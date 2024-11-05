import argparse
import torch
from PIL import Image
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from widowx_model import create_model
import yaml
import mediapy
from scipy.spatial.transform import Rotation as R

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                        help='Name or path to the pretrained model')
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    parser.add_argument('--env_name', type=str, default="widowx_spoon_on_towel", 
                        help='Name of env')
    parser.add_argument('--ctrl_freq', type=int, default=25, 
                        help='The control frequency of the robot')
    parser.add_argument('--chunk_size', type=int, default=64, 
                        help='Action chunk size')
    return parser.parse_args()

def make_policy(args):
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    model = create_model(
        args=config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    return model

def rotation_matrix_to_ortho6d_1d(matrix: np.array) -> np.array:
    """
    The orhto6d represents the first two column vectors a1 and a2 of the
    rotation matrix: [ | , |,  | ]
                     [ a1, a2, a3]
                     [ | , |,  | ]
    Input: (3, 3)
    Output: (6,)
    """
    ortho6d = matrix[:, :2]
    ortho6d = ortho6d.T
    ortho6d = ortho6d.reshape([6])
    return ortho6d

def ortho6d_to_rotation_matrix(ortho6d):
    """ Convert ortho6d (shape (6,)) to a rotation matrix (3x3). """
    return R.from_euler('xyz', ortho6d[:3]).as_matrix()


def model_inference(args, env):
    frames = []
    policy = make_policy(args)
    
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    args.lang_instruction = lang_dict['instruction']
    args.lang_name = lang_dict['name']
    lang_embeddings = lang_dict["embeddings"]
    
    obs, reset_info = env.reset()

    print("Reset info", reset_info)
    
    done, truncated = False, False
    t = 0
    action_buffer = np.zeros([args.chunk_size, env.action_space.shape[0]])
    """
    obs.keys(): ['agent', 'extra', 'camera_param', 'image']
    obs['agent'].keys(): ['qpos', 'qvel', 'controller', 'base_pose']
    tcp_pose: 3 pos + 4 quat(wxyz)
    """
    
    env.env.env.env.agent.controller.controllers['arm'].config.use_delta = False # absolute control
    env.env.env.env.agent.controller.controllers['arm'].config.frame = 'base' # absolute control
    
    while not (done or truncated):
        if t % args.chunk_size == 0:
            images = [None, None, None,get_image_from_maniskill2_obs_dict(env, obs), None, None]
            images = [Image.fromarray(img) if img is not None else None for img in images]
            proprio = torch.from_numpy(obs['agent']['qpos']).float().cuda().unsqueeze(0)
            proprio = torch.cat([proprio, torch.from_numpy(obs['agent']['qvel']).float().cuda().unsqueeze(0)], dim=1)
            tcp_pose = env.env.env.env.tcp.pose # by default, tcp pose (eef pose) world coordinate
            tcp_pose = env.env.env.env.agent.robot.pose.inv() * tcp_pose # eef pose at base see https://github.com/simpler-env/SimplerEnv/blob/d55e19162be86794875839725fd484b768e25873/tools/sysid/sysid.py#L51
            eef_xyz = tcp_pose.p # see https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/structs/pose.py#L94

            proprio = torch.cat([proprio, torch.from_numpy(eef_xyz).float().cuda().unsqueeze(0)], dim=1)
            # quat = tcp_pose[3:]
            quat = tcp_pose.q

            rr = R.from_quat(quat, scalar_first=True)
            rotmat = rr.as_matrix()
            ortho6d = rotation_matrix_to_ortho6d_1d(rotmat)


            proprio = torch.cat([proprio, torch.from_numpy(ortho6d).float().cuda().unsqueeze(0)], dim=1)

            action_buffer = policy.step(
                proprio=proprio,
                images=images,
                text_embeds=lang_embeddings
            ).squeeze(0).cpu().numpy()

        # absolute control
        action = action_buffer[t % args.chunk_size]
        gripper_action = action[-1]
        out_eef_xyz = action[:3]
        out_ortho6d = action[3:-1]
        out_rot_matrix = ortho6d_to_rotation_matrix(out_ortho6d)
        out_r = R.from_matrix(out_rot_matrix)
        out_axis_angle = out_r.as_rotvec()
  
        action = np.concatenate([out_eef_xyz, out_axis_angle, [gripper_action]])
        # action[3:] = action[3:] * 0.01
        # action[:3] = action[:3] * 0.1
        print(f"action={action}")
        
        obs, reward, done, truncated, info = env.step(action)

        frames.append(get_image_from_maniskill2_obs_dict(env, obs))
        new_instruction = env.get_language_instruction()
        if new_instruction != args.lang_instruction:
            args.lang_instruction = new_instruction
            print("New Instruction", args.lang_instruction)
        
        t += 1
        # print("Step", t)
    
    episode_stats = info.get('episode_stats', {})
    print("Episode stats", episode_stats)
    return frames

def main():
    args = get_arguments()
    env_name = args.env_name
    env = simpler_env.make(env_name)

    # env = simpler_env.make('widowx_spoon_on_towel')
    # env = simpler_env.make('widowx_carrot_on_plate')
    # env = simpler_env.make('widowx_stack_cube')
    # env = simpler_env.make('widowx_put_eggplant_in_basket')
    
    frames = model_inference(args, env)

    model_name = args.pretrained_model_name_or_path.split("/")[-1]
    lang_name = args.lang_name
    save_path = f"outs/{model_name}_{lang_name}_{args.ctrl_freq}_chunk_{args.chunk_size}.mp4"
    mediapy.write_video(save_path, frames, fps=10)
    print("save at ", end="")
    print(save_path)

if __name__ == '__main__':
    main()
