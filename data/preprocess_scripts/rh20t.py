import tensorflow as tf
import tensorflow_datasets as tfds
from data.utils import clean_task_instruction, quaternion_to_rotation_matrix_wo_static_check, \
    rotation_matrix_to_ortho6d_1d
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

def decode_image(image_data):
    image_data = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image

def _parse_function(proto):
    feature_description = {
        'joint': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'instruction': tf.io.FixedLenFeature([], tf.string),
        'terminate_episode': tf.io.FixedLenFeature([], tf.int64),
        'gripper': tf.io.FixedLenFeature([], tf.string, default_value=""), 
        'tcp': tf.io.FixedLenFeature([], tf.string, default_value=""),
        'tcp_base': tf.io.FixedLenFeature([], tf.string, default_value="")
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    
    parsed_features['joint'] = tf.io.parse_tensor(parsed_features['joint'], out_type=tf.float64)
    parsed_features['image'] = tf.io.parse_tensor(parsed_features['image'], out_type=tf.string)
    parsed_features['instruction'] = tf.io.parse_tensor(parsed_features['instruction'], out_type=tf.string)
    parsed_features['gripper'] = tf.cond(
        tf.math.equal(parsed_features['gripper'], ""),
        lambda: tf.constant([], dtype=tf.float64),
        lambda: tf.reshape(tf.io.parse_tensor(parsed_features['gripper'], out_type=tf.float64), [3])
    )
    parsed_features['tcp_base'] = tf.cond(
        tf.math.equal(parsed_features['tcp_base'], ""),
        lambda: tf.constant([], dtype=tf.float64),
        lambda: tf.reshape(tf.io.parse_tensor(parsed_features['tcp_base'], out_type=tf.float64),[7])
    )
    
    image = tf.image.decode_jpeg(parsed_features['image'],channels=3)
    joint = parsed_features['joint']
    shape = tf.shape(joint)[0]
    joint = tf.reshape(joint, [shape])

    instruction = parsed_features['instruction']
    terminate_episode = tf.cast(parsed_features['terminate_episode'],tf.int64)
    gripper = parsed_features['gripper']
    tcp_base = parsed_features['tcp_base']
    return {
        'joint': joint,
        'observation':{
            'image': image,
        },
        'instruction': instruction,
        'terminate_episode': terminate_episode,
        'gripper': gripper,
        'tcp_base': tcp_base,
    }

def dataset_generator_from_tfrecords(seed):
    tfrecord_path = './data/datasets/rh20t/tfrecords/'
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
        num = 0
        for data in dataset:
            num = num + 1
            if data['joint'].shape != tf.TensorShape((6,)) and data['joint'].shape != tf.TensorShape((7,)):
                num = 0
                break
        if num <= 1: # discard dataset
            continue
        yield {
            'steps': dataset
        }
        
def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(tf.equal(terminate_act, tf.constant(0.0, dtype=tf.int64)),tf.constant(False),tf.constant(True))


def load_dataset(seed):
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator_from_tfrecords(seed),
        output_signature={
            'steps': tf.data.DatasetSpec(
                element_spec={
                    'joint': tf.TensorSpec(shape=(None), dtype=tf.float64),
                    'observation':{
                    'image': tf.TensorSpec(shape=(None), dtype=tf.uint8),
                    },
                    'instruction': tf.TensorSpec(shape=(), dtype=tf.string),
                    'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.int64),
                    'gripper': tf.TensorSpec(shape=(None), dtype=tf.float64),
                    'tcp_base': tf.TensorSpec(shape=(None), dtype=tf.float64),
                }
            )
        }
    )

    return dataset

def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(tf.equal(terminate_act, tf.constant(0, dtype=tf.int64)),tf.constant(False),tf.constant(True))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
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
    
    
    # Convert raw action to our action
    step['action'] = {}
    step['action']['terminate'] = terminate_act_to_bool(step['terminate_episode'])

    state = step['observation']
    joint_pos = step['joint']
    state['arm_concat'] = joint_pos
    
    state_format = ""
    state_format = tf.cond(
        tf.equal(tf.shape(joint_pos), tf.shape(tf.zeros((7,), dtype=tf.float64))),
        lambda: tf.constant("arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos"),
        lambda: state_format
    )
    state_format = tf.cond(
        tf.equal(tf.shape(joint_pos), tf.shape(tf.zeros((6,), dtype=tf.float64))),
        lambda: tf.constant("arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos"),
        lambda: state_format
    )
        
    # eef
    eef = step['tcp_base']
    state['arm_concat'] = tf.cond(
        tf.equal(tf.shape(eef), tf.shape(tf.zeros((7,), dtype=tf.float64))),
        lambda: tf.concat([
            state['arm_concat'], 
            tf.concat([
                eef[:3], rotation_matrix_to_ortho6d_1d(
                    quaternion_to_rotation_matrix_wo_static_check(eef[3:]))], axis=0)], axis=0),
        lambda: state['arm_concat']
    )
    state_format = tf.cond(
        tf.equal(tf.shape(eef), tf.shape(tf.zeros((7,), dtype=tf.float64))),
        lambda: tf.strings.join([state_format, tf.constant("eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")],separator = ','),
        lambda: state_format
    )
    
    # gripper
    state['arm_concat'] = tf.cond(
        tf.equal(tf.shape(step['gripper']), tf.shape(tf.zeros((3,), dtype=tf.float64))),
        lambda: tf.concat([state['arm_concat'], step['gripper'][0:1] / 110], axis=0),
        lambda: state['arm_concat']
    )
    state_format = tf.cond(
        tf.equal(tf.shape(step['gripper']), tf.shape(tf.zeros((3,), dtype=tf.float64))),
        lambda: tf.strings.join([state_format, tf.constant("gripper_joint_0_pos")],separator = ','),
        lambda: state_format
    )
    
    state['arm_concat'] = tf.cast(state['arm_concat'], tf.float32)
    state['format'] = state_format
    
    return step

if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path
    
    # Load the dataset
    dataset = load_dataset(0)
    for data in dataset:
        for step in data['steps']:
            step = process_step(step)
            print(step['observation']['format'])
