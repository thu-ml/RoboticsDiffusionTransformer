import tensorflow as tf
from data.utils import clean_task_instruction, euler_to_rotation_matrix, rotation_matrix_to_ortho6d
import tensorflow as tf
import os
import fnmatch
import random


def _parse_function(proto):
    keys_to_features = {
        'action': tf.io.FixedLenFeature([], tf.string),
        'robot_obs': tf.io.FixedLenFeature([], tf.string),
        'rgb_static': tf.io.FixedLenFeature([], tf.string),
        'rgb_gripper': tf.io.FixedLenFeature([], tf.string),
        'terminate_episode': tf.io.FixedLenFeature([], tf.int64),
        'instruction': tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    action = tf.io.parse_tensor(parsed_features['action'], out_type=tf.float64)
    robot_obs = tf.io.parse_tensor(parsed_features['robot_obs'], out_type=tf.float64)
    rgb_static = tf.io.parse_tensor(parsed_features['rgb_static'], out_type=tf.uint8)
    rgb_gripper = tf.io.parse_tensor(parsed_features['rgb_gripper'], out_type=tf.uint8)
    instruction = parsed_features['instruction']
    terminate_episode = tf.cast(parsed_features['terminate_episode'], tf.int64)
    
    action = tf.reshape(action, [7])
    action = tf.cast(action, tf.float32)
    robot_obs = tf.reshape(robot_obs, [15])
    robot_obs = tf.cast(robot_obs, tf.float32)
    rgb_static = tf.reshape(rgb_static, [200, 200, 3])
    rgb_gripper = tf.reshape(rgb_gripper, [84, 84, 3])
    # RGB to BGR
    # rgb_static = rgb_static[:, :, ::-1]
    # rgb_gripper = rgb_gripper[:, :, ::-1]
    
    return {
        'action': action,
        'observation':{
            'robot_obs': robot_obs,
            'rgb_static': rgb_static,
            'rgb_gripper': rgb_gripper,  
        },
        'instruction': instruction,
        'terminate_episode': terminate_episode
    }


def dataset_generator_from_tfrecords(seed):
    tfrecord_path = './data/datasets/calvin/tfrecords/'
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
                    'action': tf.TensorSpec(shape=(7,), dtype=tf.float32),
                    'observation':{
                        'robot_obs': tf.TensorSpec(shape=(15,), dtype=tf.float32),
                        'rgb_static': tf.TensorSpec(shape=(200,200,3), dtype=tf.uint8),
                        'rgb_gripper': tf.TensorSpec(shape=(84,84,3), dtype=tf.uint8),
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
    return tf.where(
        tf.equal(terminate_act, tf.constant(0, dtype=tf.int64)),
        tf.constant(False),tf.constant(True))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    old_action = step['action']
    step['action'] = {}
    action = step['action']
    step['action']['terminate'] = terminate_act_to_bool(step['terminate_episode'])
    # ['actions']
    # (dtype=np.float32, shape=(7,))
    # tcp position (3): x,y,z in absolute world coordinates
    # tcp orientation (3): euler angles x,y,z in absolute world coordinates
    # gripper_action (1): binary (close = -1, open = 1)
    eef_pos = old_action[:3]
    eef_ang = euler_to_rotation_matrix(old_action[3:6])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    gripper_open = (old_action[6] + 1) / 2
    gripper_open = tf.expand_dims(gripper_open, axis=0)
    
    # # No base found
    arm_action = tf.concat([eef_pos, eef_ang, gripper_open], axis=0)
    action['arm_concat'] = arm_action
    # # Write the action format
    action['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,gripper_open")
    
    state = step['observation']
    # ['robot_obs']
    # (dtype=np.float32, shape=(15,))
    # tcp position (3): x,y,z in world coordinates
    # tcp orientation (3): euler angles x,y,z in world coordinates
    # gripper opening width (1): in meter
    # arm_joint_states (7): in rad
    # gripper_action (1): binary (close = -1, open = 1)
    eef_pos = state['robot_obs'][:3]
    eef_ang = euler_to_rotation_matrix(state['robot_obs'][3:6])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    gripper_open = (state['robot_obs'][14] + 1) / 2
    gripper_open = tf.expand_dims(gripper_open, axis=0)
    qpos = state['robot_obs'][7:14]
    
    state['arm_concat'] = tf.concat([qpos,gripper_open,eef_pos,eef_ang], axis=0)
    # # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
    instr=  clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    # Load the dataset
    dataset = load_dataset(1717055919)
    for data in dataset.take(1):
        for step in data['steps']:
            step = process_step(step)
            print(step['observation']['natural_language_instruction'])
