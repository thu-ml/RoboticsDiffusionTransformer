import tensorflow as tf
import tensorflow_datasets as tfds
from data.utils import clean_task_instruction, quaternion_to_euler, euler_to_quaternion, euler_to_rotation_matrix, rotation_matrix_to_ortho6d
import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm
import os
import imageio
import concurrent.futures
import fnmatch
import random


def _parse_function(proto):
    keys_to_features = {
        'observations/images0': tf.io.FixedLenFeature([], tf.string),
        'observations/state': tf.io.FixedLenFeature([], tf.string),
        'observations/qpos': tf.io.FixedLenFeature([], tf.string),
        'observations/eef_transform': tf.io.FixedLenFeature([], tf.string),
        'language': tf.io.FixedLenFeature([], tf.string),
        'actions': tf.io.FixedLenFeature([], tf.string),
        'truncates': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    observations_images0 = tf.io.parse_tensor(parsed_features['observations/images0'], out_type=tf.uint8)
    observations_state = tf.io.parse_tensor(parsed_features['observations/state'], out_type=tf.float32)
    observations_qpos = tf.io.parse_tensor(parsed_features['observations/qpos'], out_type=tf.float32)
    # observations_eef_transform = tf.io.parse_tensor(parsed_features['observations/eef_transform'], out_type=tf.float32)
    language = parsed_features['language']
    actions = tf.io.parse_tensor(parsed_features['actions'], out_type=tf.float32)
    truncates = parsed_features['truncates']
    
    actions = tf.reshape(actions, [7])
    observations_images0 = tf.reshape(observations_images0, [480, 640, 3])
    # observations_eef_transform = tf.reshape(observations_eef_transform, [4,4])
    # observations_eef_transform = extract_angles_and_translation(observations_eef_transform)
    # observations_eef_transform = tf.reshape(observations_eef_transform, [6])
    observations_qpos = tf.reshape(observations_qpos, [6])
    observations_state = tf.reshape(observations_state, [7])
    
    return {
        'observation': {
            'images0': observations_images0,
            'state': observations_state,
            'qpos': observations_qpos,
        },
        'language': language,
        'actions': actions,
        'truncates': truncates
    }


def dataset_generator_from_tfrecords(seed):
    tfrecord_path = './data/datasets/bridgev2/tfrecords'
    filepaths = []
    for root, dirs, files in os.walk(tfrecord_path):
        for filename in fnmatch.filter(files, '*.tfrecord'):
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)
            
    random.seed(seed)
    random.shuffle(filepaths)
    for filepath in filepaths:
        raw_dataset = tf.data.TFRecordDataset(filepath)
        dataset = raw_dataset.map(_parse_function)
        yield {
            'steps': dataset
        }
        
def load_dataset(seed):
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator_from_tfrecords(seed),
        output_signature={
            'steps': tf.data.DatasetSpec(
                element_spec={
                    'observation': {
                        'images0': tf.TensorSpec(shape=(480, 640, 3), dtype=tf.uint8),
                        'state': tf.TensorSpec(shape=(7,), dtype=tf.float32),
                        'qpos': tf.TensorSpec(shape=(6,), dtype=tf.float32),
                    },
                    'language': tf.TensorSpec(shape=(), dtype=tf.string),
                    'actions': tf.TensorSpec(shape=(7,), dtype=tf.float32),
                    'truncates': tf.TensorSpec(shape=(), dtype=tf.int64)
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
    old_action = step['actions']
    step['action'] = {}
    action = step['action']
    step['action']['terminate'] = step['truncates']
    # https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/utils/transformation_utils.py line 154
    eef_delta_pos = old_action[:3]
    eef_ang = old_action[3:6]
    eef_ang = euler_to_quaternion(eef_ang)
    gripper_state = old_action[6]
    # https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/base/robot_base_env.py line 231
    # gripper_open = tf.constant(0.0,dtype=tf.float32) if gripper_state < 0.5 else tf.constant(1.0,dtype=tf.float32)
    gripper_open = tf.cond(tf.less(gripper_state, 0.5), lambda: tf.constant(0.0, dtype=tf.float32), lambda: tf.constant(1.0, dtype=tf.float32))
    gripper_open = tf.expand_dims(gripper_open,axis=0)

    # # No base found
    # # Concatenate the action
    arm_action = tf.concat([eef_delta_pos, eef_ang,gripper_open], axis=0)
    action['arm_concat'] = arm_action
    # # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z, eef_delta_angle_x, eef_delta_angle_y, eef_delta_angle_z, eef_delta_angle_w, gripper_open")

    old_state = step['observation']['state']
    qpos = step['observation']['qpos']
    state = step['observation']

    # https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/base/robot_base_env.py line 292
    eef_pos = old_state[:3]
    eef_ang = old_state[3:6]
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    gripper_open = old_state[6:]
    # gripper_open = tf.cond(tf.less(gripper_state, 0.5), lambda: tf.constant(0.0, dtype=tf.float32), lambda: tf.constant(1.0, dtype=tf.float32))
    # gripper_open = tf.expand_dims(gripper_open,axis=0)

    state['arm_concat'] = tf.concat([qpos,gripper_open,eef_pos,eef_ang], axis=0)
    # # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,gripper_joint_0_pos,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
    instr = step['language']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    # Load the dataset
    dataset = load_dataset(0)
    for episode in dataset.take(1):
        for step in episode['steps']:
            step = process_step(step)
            print(step)
            break

