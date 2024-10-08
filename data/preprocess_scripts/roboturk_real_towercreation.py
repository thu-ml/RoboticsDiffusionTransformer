import tensorflow as tf
import tensorflow_datasets as tfds
from data.utils import clean_task_instruction, quaternion_to_euler
import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm
import os
import imageio
import concurrent.futures

def get_frames(file_path):
    if not os.path.exists(file_path) or not os.path.isfile(file_path) or not file_path.endswith('.mp4'):
        return []
    frames = []
    with imageio.get_reader(file_path, 'ffmpeg') as reader:
        for frame in reader:
            frame = np.array(frame, dtype=np.uint8) 
            frames.append(frame)
    return frames

def parallel_get_frames(paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(get_frames, path): path for path in paths}
        return [future.result() for future in concurrent.futures.as_completed(future_to_path)]

def count_total_samples(filename):
    total_samples = 0
    with h5py.File(filename, 'r') as f:
        data = f['data']
        for user_key in data.keys():
            user = data[user_key]
            for demo_key in user.keys():
                total_samples += 1
    return total_samples

def dataset_generator(filename, total_samples):
    with h5py.File(filename, 'r') as f:
        data = f['data']
        for user_key in data.keys():
            user = data[user_key]
            for demo_key in user.keys():
                demo = user[demo_key]
                robot_observation = demo['robot_observation']
                user_control = demo['user_control']

                eef_poses = robot_observation['eef_poses']
                joint_states_arm = robot_observation['joint_states_arm']
                joint_states_gripper = robot_observation['joint_states_gripper']
                user_control_data = user_control['user_control']

                attrs = dict(demo.attrs)
                top_depth_video_file = attrs['top_depth_video_file']
                top_rgb_video_file = attrs['top_rgb_video_file']
                front_rgb_video_file = attrs['front_rgb_video_file']

                video_root_path = './data/datasets/roboturk/'
                top_depth_frames = get_frames(os.path.join(video_root_path, top_depth_video_file))
                top_rgb_frames = get_frames(os.path.join(video_root_path, top_rgb_video_file))
                front_rgb_frames = get_frames(os.path.join(video_root_path, front_rgb_video_file))
                
                if len(top_rgb_frames) == 0 or len(front_rgb_frames) == 0:
                    continue
                # video_root_path = '/cephfs-thu/gsm_data/robotruck'
                # video_paths = [
                #     os.path.join(video_root_path, attrs['top_depth_video_file']),
                #     os.path.join(video_root_path, attrs['top_rgb_video_file']),
                #     os.path.join(video_root_path, attrs['front_rgb_video_file'])
                # ]
                # top_depth_frames, top_rgb_frames, front_rgb_frames = parallel_get_frames(video_paths)

                steps = []
                for i in range(len(eef_poses)):
                    task_demo_id = f"SawyerTowerCreation_{demo_key}_{i}"
                    step = {
                        'task_demo_id': task_demo_id,
                        'eef_poses': eef_poses[i],
                        'joint_states_arm': joint_states_arm[i],
                        'joint_states_gripper': joint_states_gripper[i],
                        'user_control': user_control_data[i] if user_control_data.shape[0] > 0 else np.zeros(22),
                        'observation':{
                            'top_depth_frame': top_depth_frames[i] if i < len(top_depth_frames) else np.zeros((0,0, 3), dtype=np.uint8),
                            'top_rgb_frame': top_rgb_frames[i] if i < len(top_rgb_frames) else np.zeros((0, 0, 3), dtype=np.uint8),
                            'front_rgb_frame': front_rgb_frames[i] if i < len(front_rgb_frames) else np.zeros((0, 0, 3), dtype=np.uint8),
                        },
                        'terminate_episode': i == len(eef_poses) - 1
                    }
                    steps.append(step)
                    

                steps_dataset = tf.data.Dataset.from_generator(
                    lambda: iter(steps),
                    output_signature={
                        'task_demo_id': tf.TensorSpec(shape=(), dtype=tf.string),
                        'eef_poses': tf.TensorSpec(shape=(7,), dtype=tf.float32),
                        'joint_states_arm': tf.TensorSpec(shape=(27,), dtype=tf.float32),
                        'joint_states_gripper': tf.TensorSpec(shape=(3,), dtype=tf.float32),
                        'user_control': tf.TensorSpec(shape=(22,), dtype=tf.float32),
                        'observation':{
                            'top_depth_frame': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                            'top_rgb_frame': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                            'front_rgb_frame': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                        },
                        'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.bool),
                    }
                )

                yield {'steps': steps_dataset}                    

def load_dataset():
    filename = './data/datasets/roboturk/SawyerTowerCreation_aligned_dataset.hdf5'
    total_samples = count_total_samples(filename)
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(filename, total_samples),
        output_signature={
            'steps': tf.data.DatasetSpec(
                element_spec={
                    'task_demo_id': tf.TensorSpec(shape=(), dtype=tf.string), 
                    'eef_poses': tf.TensorSpec(shape=(7,), dtype=tf.float32),
                    'joint_states_arm': tf.TensorSpec(shape=(27,), dtype=tf.float32),
                    'joint_states_gripper': tf.TensorSpec(shape=(3,), dtype=tf.float32),
                    'user_control': tf.TensorSpec(shape=(22,), dtype=tf.float32),
                    'observation':{
                        'top_depth_frame': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                        'top_rgb_frame': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                        'front_rgb_frame': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                    },
                    'terminate_episode': tf.TensorSpec(shape=(), dtype = tf.bool),
                }
            )
        }
    )
    return dataset

def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(tf.equal(terminate_act, tf.constant(0.0, dtype=tf.float32)),tf.constant(False),tf.constant(True))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    step['action'] = {}
    action = step['action']
    action['terminate'] = step['terminate_episode']

    eef_delta_pos = step['eef_poses'][:3]
    eef_ang = quaternion_to_euler(step['eef_poses'][3:])

    # No base found
    # Concatenate the action
    arm_action = tf.concat([eef_delta_pos, eef_ang], axis=0)
    action['arm_concat'] = arm_action

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_roll,eef_delta_angle_pitch,eef_delta_angle_yaw")

    # No state found
    state = step['observation']
    # joint_states_arm: dataset of (num_timestamps, 27) shape where each of the 9 joints is represented by the JointState message
    # (the nine joints are in order by their ROSBAG names: ['head_pan', 'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'torso_t0']. For the most part, head_pan and torso should be zeros)
    # [0] the position of the first joint (rad or m)
    # [1] the velocity of the first joint (rad/s or m/s)
    # [2] the effort that is applied in the first joint
    # [3] the position of the second joint...
    joint_states_arm = step['joint_states_arm']
    joint_pos = joint_states_arm[3:24:3]  
    joint_vel = joint_states_arm[4:25:3] 
    # joint_states_gripper: dataset of (num_timestamps, 3) shape
    # [0] the position of the gripper (rad or m)
    # [1] the velocity of the gripper (rad/s or m/s)
    # [2] the effort that is applied in the gripper
    joint_states_gripper = step['joint_states_gripper']
    gripper_pos = joint_states_gripper[:1]
    # remove gripper_vel due to they are all zeros
    # gripper_vel = joint_states_gripper[1:2]
    # Concatenate the state
    # state['arm_concat'] = tf.concat([joint_pos,joint_vel,gripper_pos,gripper_vel], axis=0)
    state['arm_concat'] = tf.concat([joint_pos,joint_vel,gripper_pos], axis=0)
    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_joint_0_pos")

    # Clean the task instruction
    # Define the replacements (old, new) as a dictionary
    replacements = {
        '_': ' ',
        '1f': ' ',
        '4f': ' ',
        '-': ' ',
        '50': ' ',
        '55': ' ',
        '56': ' ',
        
    }
    # copied from openxembod
    instr = b'create tower'
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = '/cephfs-thu/gsm_data/openx_embod'
    DATASET_NAME = 'roboturk_real_laundrylayout'
    # Load the dataset
    dataset = load_dataset()
    
    # save_dir = os.path.join(DATASET_DIR, DATASET_NAME)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # tf.data.experimental.save(dataset, save_dir)
