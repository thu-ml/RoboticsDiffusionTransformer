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
from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng
import imageio
import jax
import jax.numpy as jnp
from octo.utils.train_callbacks import supply_rng
from functools import partial

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

env_id = args.env_id
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

def sample_actions(
    pretrained_model: OctoModel,
    observations,
    tasks,
    rng,
):
    # add batch dim to observations
    observations = jax.tree_map(lambda x: x[None], observations)
    actions = pretrained_model.sample_actions(
        observations,
        tasks,
        rng=rng,
    )
    # remove batch dim
    return actions[0]

pretrain_path = args.pretrained_path
step = 1000000
model = OctoModel.load_pretrained(
        pretrain_path,
        step
    )

policy = supply_rng(
    partial(
        sample_actions,
        model,
    )
)


import tensorflow as tf
def resize_img(image, size=(256, 256)):
    image_tf = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tf = tf.expand_dims(image_tf, axis=0)
    resized_tf = tf.image.resize(
        image_tf, 
        size, 
        method=tf.image.ResizeMethod.LANCZOS3, 
        antialias=True
    )
    resized_tf = tf.squeeze(resized_tf)
    resized_img = resized_tf.numpy().astype(np.uint8)
    return resized_img

MAX_EPISODE_STEPS = 400
total_episodes = args.num_traj 
success_count = 0 
base_seed = 20241201
import tqdm

for episode in tqdm.trange(total_episodes):
    task = model.create_tasks(texts=[task2lang[env_id]])
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = base_seed)

    img = env.render().squeeze(0).detach().cpu().numpy()
    proprio = obs['agent']['qpos'][:]
    obs_window.append({
        'proprio': proprio.detach().cpu().numpy(),
        "image_primary": resize_img(img)[None],
        "timestep_pad_mask": np.zeros((1),dtype = bool)
    })
    obs_window.append({
        'proprio': proprio.detach().cpu().numpy(),
        "image_primary": resize_img(img)[None],
        "timestep_pad_mask": np.ones((1),dtype = bool)
    })
    
    global_steps = 0
    video_frames = []

    success_time = 0
    done = False

    while global_steps < MAX_EPISODE_STEPS and not done:
        obs = {
            'proprio': np.concatenate([obs_window[0]['proprio'], obs_window[1]['proprio']], axis=0),
            "image_primary": np.concatenate([obs_window[0]['image_primary'], obs_window[1]['image_primary']], axis=0),
            "timestep_pad_mask": np.concatenate([obs_window[0]['timestep_pad_mask'], obs_window[1]['timestep_pad_mask']], axis=0)
        }        
        actions = policy(obs, task)
        actions = jax.device_put(actions, device=jax.devices('cpu')[0])
        actions = jax.device_get(actions)
        # actions = actions[0:4]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            img = env.render().squeeze(0).detach().cpu().numpy()
            proprio = obs['agent']['qpos'][:]
            obs_window.append({
                'proprio': proprio.detach().cpu().numpy(),
                "image_primary": resize_img(img)[None],
                "timestep_pad_mask": np.ones((1),dtype = bool)
            })
            video_frames.append(img)
            global_steps += 1
            if terminated or truncated:
                assert "success" in info, sorted(info.keys())
                if info['success']:
                    done = True
                    success_count += 1
                    break 
    print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")

success_rate = success_count / total_episodes * 100
print(f"Random seed: {seed}, Pretrained_path: {pretrain_path}")
print(f"Tested {total_episodes} episodes, success rate: {success_rate:.2f}%")
log_file = "results_octo.log"
with open(log_file, 'a') as f:
    f.write(f"{seed}:{success_count}\n")
