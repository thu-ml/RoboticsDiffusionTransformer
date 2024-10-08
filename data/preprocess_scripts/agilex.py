import json
import random
import os
import fnmatch
import random

import tensorflow as tf


def _parse_function(proto, precomputed_instr_embed_path):
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
    cam_high = tf.io.parse_tensor(parsed_features['cam_high'], out_type=tf.string)
    cam_left_wrist = tf.io.parse_tensor(parsed_features['cam_left_wrist'], out_type=tf.string)
    cam_right_wrist = tf.io.parse_tensor(parsed_features['cam_right_wrist'], out_type=tf.string)
    # instruction = parsed_features['instruction']
    terminate_episode = tf.cast(parsed_features['terminate_episode'], tf.int64)
    
    cam_high = tf.image.decode_jpeg(cam_high, channels=3, dct_method='INTEGER_ACCURATE')
    cam_left_wrist = tf.image.decode_jpeg(cam_left_wrist, channels=3, dct_method='INTEGER_ACCURATE')
    cam_right_wrist = tf.image.decode_jpeg(cam_right_wrist, channels=3, dct_method='INTEGER_ACCURATE')
    # BGR to RGB
    cam_high = tf.reverse(cam_high, axis=[-1])
    cam_left_wrist = tf.reverse(cam_left_wrist, axis=[-1])
    cam_right_wrist = tf.reverse(cam_right_wrist, axis=[-1])

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
        "instruction": precomputed_instr_embed_path,
        "terminate_episode": terminate_episode,
    }


def dataset_generator_from_tfrecords(seed):
    tfrecord_path = './data/datasets/agilex/tfrecords/'
    filepaths = []
    for root, dirs, files in os.walk(tfrecord_path):
        for filename in fnmatch.filter(files, '*.tfrecord'):
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)
    
    random.seed(seed)
    random.shuffle(filepaths)
    for filepath in filepaths:
        raw_dataset = tf.data.TFRecordDataset(filepath)
        dataset = raw_dataset.map(lambda x: _parse_function(x, os.path.dirname(filepath)))
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
                    'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.int64),
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
    left_gripper_open = old_action[6:7] / 11.8997
    right_arm_pos = old_action[7:13]
    right_gripper_open = old_action[13:14] / 13.9231

    # Base action is dummy (all zeros)
    arm_action = tf.concat([left_arm_pos,left_gripper_open,right_arm_pos,right_gripper_open], axis=0)
    action['arm_concat'] = arm_action
    # Write the action format
    action['format'] = tf.constant(
        "left_arm_joint_0_pos,left_arm_joint_1_pos,left_arm_joint_2_pos,left_arm_joint_3_pos,left_arm_joint_4_pos,left_arm_joint_5_pos,left_gripper_open,right_arm_joint_0_pos,right_arm_joint_1_pos,right_arm_joint_2_pos,right_arm_joint_3_pos,right_arm_joint_4_pos,right_arm_joint_5_pos,right_gripper_open")
    
    state = step['observation']
    left_qpos = step['qpos'][:6]
    left_gripper_open = step['qpos'][6:7] / 4.7908     # rescale to [0, 1]
    right_qpos = step['qpos'][7:13]
    right_gripper_open = step['qpos'][13:14] / 4.7888  # rescale to [0, 1]
    state['arm_concat'] = tf.concat([left_qpos, left_gripper_open,right_qpos, right_gripper_open], axis=0)
    # # Write the state format
    state['format'] = tf.constant(
        "left_arm_joint_0_pos,left_arm_joint_1_pos,left_arm_joint_2_pos,left_arm_joint_3_pos,left_arm_joint_4_pos,left_arm_joint_5_pos,left_gripper_open,right_arm_joint_0_pos,right_arm_joint_1_pos,right_arm_joint_2_pos,right_arm_joint_3_pos,right_arm_joint_4_pos,right_arm_joint_5_pos,right_gripper_open")

    # We randomly sample [original,expanded,simplified] instructions. The ratio is 1:1:1
    instr_type = tf.random.uniform(shape=(), minval=0, maxval=3, dtype=tf.int32)
    # # NOTE bg : tf.random and tf.constant is buggy as it always return 0 (?)
    # instr_type = tf.constant(instr_type)
    # print(instr_type)
    @tf.function
    def f0(): 
        return tf.strings.join([step['instruction'], tf.constant('/lang_embed_0.pt')])
    @tf.function
    def f1(): 
        return tf.strings.join([step['instruction'], tf.constant('/lang_embed_1.pt')])
    @tf.function
    def f2():
        index = tf.random.uniform(shape=(), minval=0, maxval=100, dtype=tf.int32)
        return tf.strings.join([
            step['instruction'], tf.constant('/lang_embed_'), tf.strings.as_string(index+2), tf.constant('.pt')])

    instr = tf.case([
        (tf.equal(instr_type, 0), f0),
        (tf.equal(instr_type, 1), f1),
        (tf.equal(instr_type, 2), f2)
    ], exclusive=True)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":  
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    dataset = load_dataset(42)
    
    for episode in dataset:
        for step in episode['steps']:
            step = process_step(step)
            # save the images
            print(step['observation']['natural_language_instruction'])
