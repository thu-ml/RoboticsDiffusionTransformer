from typing import Callable, List, Type
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
import torch
from collections import deque
from PIL import Image
import cv2
import imageio
from functools import partial
from torchvision.transforms.functional import center_crop

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    return parser.parse_args()

task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

import random
import os

args = parse_args()
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import AutoModelForVision2Seq, AutoProcessor

DATA_STAT = {'mean': [ 0.00263866,  0.01804881, -0.02151551, -0.00384866,  0.00500441,                                                                                                            
       -0.00057146, -0.26013601], 'std': [0.06639539, 0.1246438 , 0.09675793, 0.03351422, 0.04930534,                                                                                           
       0.25787726, 0.96762997], 'max': [0.31303197, 0.77948809, 0.42906255, 0.20186238, 0.63990456,                                                                                             
       0.99999917, 1.        ], 'min': [-0.31464151, -0.64183694, -0.62718982, -0.5888508 , -0.97813392,                                                                                     
       -0.99999928, -1.        ], 'q01': [-0.18656027, -0.31995443, -0.24702898, -0.18005923, -0.2164692 ,                                                                                      
       -0.82366071, -1.        ], 'q99': [0.18384692, 0.45547636, 0.27452313, 0.03571117, 0.1188747 ,                                                                                           
       0.85074112, 1.        ]}

MODEL_PATH = args.pretrained_path

def make_policy():
    device = torch.device('cuda')
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)
    vla.norm_stats["maniskill"] = {
        "action": {
            "min": np.array(DATA_STAT["min"]),
            "max": np.array(DATA_STAT["max"]),
            "mean": np.array(DATA_STAT["mean"]),
            "std": np.array(DATA_STAT["std"]),
            "q01": np.array(DATA_STAT["q01"]),
            "q99": np.array(DATA_STAT["q99"]),
        }
    }
    
    vla = vla.eval()

    return vla, processor

vla, processor = make_policy()
success_counts = {}

for env_id in task2lang.keys():
    
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_delta_pose",
        render_mode=args.render_mode,
        reward_mode="dense" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend
    )
    
    MAX_EPISODE_STEPS = 400
    total_episodes = args.num_traj 
    success_count = 0  
    base_seed = 20241201
    import tqdm

    for episode in tqdm.trange(total_episodes):
        obs_window = deque(maxlen=2)
        obs, _ = env.reset(seed = base_seed + episode)

        img = env.render().squeeze(0).detach().cpu().numpy()
        obs_window.append(img)
        
        global_steps = 0
        video_frames = []

        success_time = 0
        done = False

        while global_steps < MAX_EPISODE_STEPS and not done:
            obs = obs_window[-1]        
            image_arrs = [
                obs_window[-1]
            ]
            images = [Image.fromarray(arr) for arr in image_arrs]
            original_size = images[0].size
            crop_scale = 0.9
            sqrt_crop_scale = crop_scale
            sqrt_crop_scale = np.sqrt(crop_scale)
            images = [
                center_crop(
                    img, output_size=(
                        int(sqrt_crop_scale * img.size[1]), 
                        int(sqrt_crop_scale * img.size[0])
                    )
                ) for img in images
            ]
            images = [img.resize(original_size, Image.Resampling.BILINEAR) for img in images]
            # de-capitalize and remove trailing period
            instruction = task2lang[env_id].lower()
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
            actions = vla.predict_action(**inputs, unnorm_key="maniskill", do_sample=False)[None]
            for idx in range(actions.shape[0]):
                action = actions[idx]
                # print(action)
                # action = action * (np.array(DATA_STAT['std']) + 1e-8) + np.array(DATA_STAT['mean'])
                obs, reward, terminated, truncated, info = env.step(action)
                img = env.render().squeeze(0).detach().cpu().numpy()
                obs_window.append(img)
                video_frames.append(img)
                global_steps += 1
                if terminated or truncated:
                    assert "success" in info, sorted(info.keys())
                    if info['success']:
                        success_count += 1
                        done = True
                        break
        print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")
    success_counts[env_id] = success_count
    print(f"Task {env_id} finished, success: {success_count}/{total_episodes}")

log_file = "results_ovla_all.log"
with open(log_file, 'a') as f:
    f.write(f"{seed}:{success_counts}\n")