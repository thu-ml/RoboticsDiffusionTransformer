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
import fnmatch
import cv2
import random

def _parse_function(proto):
    keys_to_features = {
        'action': tf.io.FixedLenFeature([], tf.string),
        'base_action': tf.io.FixedLenFeature([], tf.string),
        'qpos': tf.io.FixedLenFeature([], tf.string),
        'qvel': tf.io.FixedLenFeature([], tf.string),
        'cam_high': tf.io.FixedLenFeature([], tf.string),
        'cam_left_wrist': tf.io.FixedLenFeature([], tf.string),
        'cam_right_wrist': tf.io.FixedLenFeature([], tf.string),
        'instruction': tf.io.FixedLenFeature([], tf.string),
        'terminate_episode': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    action = tf.io.parse_tensor(parsed_features['action'], out_type=tf.float32)
    base_action = tf.io.parse_tensor(parsed_features['base_action'], out_type=tf.float32)
    qpos = tf.io.parse_tensor(parsed_features['qpos'], out_type=tf.float32)
    qvel = tf.io.parse_tensor(parsed_features['qvel'], out_type=tf.float32)
    cam_high = tf.io.parse_tensor(parsed_features['cam_high'], out_type=tf.uint8)
    cam_left_wrist = tf.io.parse_tensor(parsed_features['cam_left_wrist'], out_type=tf.uint8)
    cam_right_wrist = tf.io.parse_tensor(parsed_features['cam_right_wrist'], out_type=tf.uint8)
    instruction = parsed_features['instruction']
    terminate_episode = tf.cast(parsed_features['terminate_episode'], tf.int64)
    action = tf.reshape(action, [14])
    base_action = tf.reshape(base_action, [2])
    qpos = tf.reshape(qpos, [14])
    qvel = tf.reshape(qvel, [14])
    cam_high = tf.reshape(cam_high, [480, 640, 3])
    cam_left_wrist = tf.reshape(cam_left_wrist, [480, 640, 3])
    cam_right_wrist = tf.reshape(cam_right_wrist, [480, 640, 3])

    return {
        "action": action,
        "base_action": base_action,
        "qpos": qpos,
        "qvel": qvel,
        'observation':{
        "cam_high": cam_high,
        "cam_left_wrist": cam_left_wrist,
        "cam_right_wrist": cam_right_wrist
        },
        "instruction": instruction,
        "terminate_episode": terminate_episode
    }

def dataset_generator_from_tfrecords(seed):
    tfrecord_path = './data/datasets/aloha/tfrecords/aloha_mobile/'
    datasets = []
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
                    'action': tf.TensorSpec(shape=(14), dtype=tf.float32),
                    'base_action': tf.TensorSpec(shape=(2), dtype=tf.float32),
                    'qpos': tf.TensorSpec(shape=(14), dtype=tf.float32),
                    'qvel': tf.TensorSpec(shape=(14), dtype=tf.float32),
                    'observation': {
                        'cam_high': tf.TensorSpec(shape=(480, 640, 3), dtype=tf.uint8),
                        'cam_left_wrist': tf.TensorSpec(shape=(480, 640, 3), dtype=tf.uint8),
                        'cam_right_wrist': tf.TensorSpec(shape=(480, 640, 3), dtype=tf.uint8),
                    },
                    'instruction': tf.TensorSpec(shape=(), dtype=tf.string),
                    'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.int64)
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
    old_action = step['action']
    step['action'] = {}
    action = step['action']
    step['action']['terminate'] = step['terminate_episode']
    # act-plus-plus/utils.py at main · MarkFzp/act-plus-plus
    left_arm_pos = old_action[:6]
    left_gripper_open = old_action[6:7]
    right_arm_pos = old_action[7:13]
    right_gripper_open = old_action[13:14]

    base_vel_y = step['base_action'][:1]
    base_delta_ang = step['base_action'][1:]
    base_action = tf.concat([base_vel_y, base_delta_ang], axis=0)
    # # No base found
    arm_action = tf.concat([left_arm_pos,left_gripper_open,right_arm_pos,right_gripper_open], axis=0)
    action['arm_concat'] = arm_action
    action['base_concat'] = base_action
    # # Write the action format
    action['format'] = tf.constant(
        "left_arm_joint_0_pos,left_arm_joint_1_pos,left_arm_joint_2_pos,left_arm_joint_3_pos,left_arm_joint_4_pos,left_arm_joint_5_pos,left_gripper_open,right_arm_joint_0_pos,right_arm_joint_1_pos,right_arm_joint_2_pos,right_arm_joint_3_pos,right_arm_joint_4_pos,right_arm_joint_5_pos,right_gripper_open,base_vel_y,base_angular_vel")
    
    state = step['observation']
    left_qpos = step['qpos'][:6]
    left_gripper_open = step['qpos'][6:7]
    right_qpos = step['qpos'][7:13]
    right_gripper_open = step['qpos'][13:14]
    left_qvel = step['qvel'][:6]
    # left_gripper_joint_vel = step['qvel'][6:7]
    right_qvel = step['qvel'][7:13]
    # right_gripper_joint_vel = step['qvel'][13:14]

    state['arm_concat'] = tf.concat([left_qpos, left_qvel, left_gripper_open, right_qpos, right_qvel, right_gripper_open], axis=0)
    # # Write the state format
    state['format'] = tf.constant(
        "left_arm_joint_0_pos,left_arm_joint_1_pos,left_arm_joint_2_pos,left_arm_joint_3_pos,left_arm_joint_4_pos,left_arm_joint_5_pos,left_arm_joint_0_vel,left_arm_joint_1_vel,left_arm_joint_2_vel,left_arm_joint_3_vel,left_arm_joint_4_vel,left_arm_joint_5_vel,left_gripper_open,right_arm_joint_0_pos,right_arm_joint_1_pos,right_arm_joint_2_pos,right_arm_joint_3_pos,right_arm_joint_4_pos,right_arm_joint_5_pos,right_arm_joint_0_vel,right_arm_joint_1_vel,right_arm_joint_2_vel,right_arm_joint_3_vel,right_arm_joint_4_vel,right_arm_joint_5_vel,right_gripper_open")

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
    instr = step['instruction']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = '/mnt/d/aloha/'
    DATASET_NAME = 'dataset'
    # Load the dataset
    dataset = load_dataset()
    for data in dataset.take(1):
        for step in data['steps'].take(1):
            from IPython import embed; embed()
            print(step)