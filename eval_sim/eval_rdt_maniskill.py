from typing import Callable, List, Type
import sys
sys.path.append('/')
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    return parser.parse_args()

import random
import os

# set cuda 
args = parse_args()
# set random seeds
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

env_id = args.env_id
env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    sim_backend=args.sim_backend
)

config_path = 'configs/base.yaml'
with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
pretrained_path = args.pretrained_path
policy = create_model(
    args=config, 
    dtype=torch.bfloat16,
    pretrained=pretrained_path,
    pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
)

if os.path.exists(f'text_embed_{env_id}.pt'):
    text_embed = torch.load(f'text_embed_{env_id}.pt')
else:
    text_embed = policy.encode_instruction(task2lang[env_id])
    torch.save(text_embed, f'text_embed_{env_id}.pt')

MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  
success_count = 0  

base_seed = 20241201
import tqdm
for episode in tqdm.trange(total_episodes):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + base_seed)
    policy.reset()

    img = env.render().squeeze(0).detach().cpu().numpy()
    obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs['agent']['qpos'][:, :-1]

    global_steps = 0
    video_frames = []

    success_time = 0
    done = False

    while global_steps < MAX_EPISODE_STEPS and not done:
        image_arrs = []
        for window_img in obs_window:
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        images = [Image.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
        # Take 8 steps since RDT is trained to predict interpolated 64 steps(actual 14 steps)
        actions = actions[::4, :]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            img = env.render().squeeze(0).detach().cpu().numpy()
            obs_window.append(img)
            proprio = obs['agent']['qpos'][:, :-1]
            video_frames.append(img)
            global_steps += 1
            if terminated or truncated:
                assert "success" in info, sorted(info.keys())
                if info['success']:
                    success_count += 1
                    done = True
                    break 
    print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")

success_rate = success_count / total_episodes * 100
print(f"Success rate: {success_rate}%")
