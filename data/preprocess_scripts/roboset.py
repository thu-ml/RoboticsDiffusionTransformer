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
path2json = {
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaPlanarPushReal_v2d/set_17_plannar_push_eval": [
        "Push the green object to the red line."
    ],
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaBinReorientRealRP03_v2d_set16/set_16_bin_reorient_3": [
        "Pick up the bottle and and stand it upright."
    ],
    "mnt/raid5/data/jaydv/robohive_base/demonstrations/orange_block/set_7_orange_block": [
        "Pick up the orange block."
    ],
    "mnt/raid5/data/jaydv/robohive_base/demonstrations/wooden_block": [
        "Pick up the wooden block."
    ],
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaBinReorientReal_v2d-orig/set_15_bin_reorient_eval_2": [
        "Pick up the bottle and and stand it upright."
    ],
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaBinPushReal_v2d_set14/set_14_bin_push": [
        "Push the object to the red line."
    ],
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaBinPickRealRP05_v2d/set_11_bottle_pick": [
        "Pick up the bottle."
    ],
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaBinPickRealRP03_v2d/set_10_bin_pick_2004": [
        "Pick up the wooden block."
    ],
    "mnt/raid5/data/jaydv/bin_pick_data_30029/set_2_softtoys": [
        "Pick up one toy in the basket."
    ],
    "home/jaydv/Documents/RoboSet/pick_banana_from_toaster_place_on_table_data": [
        "Pick banana from toaster and place on table."
    ],
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaBinPickReal_v2d/set_12_bin_pick_eval": [
        "Pick up the block."
    ],
    "mnt/raid5/data/roboset/v0.3/flap_open_toaster_oven_data": [
        "Flap open toaster."
    ],
    "mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaBinReorientReal_v2d_set13/set_13_bin_reorient": [
        "Pick up the bottle and stand it upright."
    ],
    "home/jaydv/Documents/RoboSet/drag_mug_from_right_to_left_data": [
        "Drag mug right to left."
    ],
    "home/jaydv/Documents/RoboSet/drag_strainer_backward_data": [
        "Drag strainer backwards."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_4/baking_prep_slide_close_drawer_scene_4": [
        "Slide and close the drawer."
    ],
    "mnt/raid5/data/roboset/v0.4/heat_soup/scene_4/baking_slide_in_bowl_scene_4": [
        "Place the bowl into the container."
    ],
    "set_5_bottle_cube_14": [
        "Pick up the bottle."
    ],
    "home/jaydv/Documents/RoboSet/drag_mug_forward_data": [
        "Drag mug forwards."
    ],
    "mnt/raid5/data/roboset/v0.4/clean_kitchen/scene_3/clean_kitchen_pick_towel_scene_3": [
        "Pick up the towel from the oven."
    ],
    "mnt/raid5/data/roboset/v0.4/heat_soup/scene_2/baking_pick_bowl_scene_2": [
        "Pick up the bowl."
    ],
    "mnt/raid5/data/roboset/v0.4/heat_soup/scene_2/baking_slide_in_bowl_scene_2": [
        "Place the bowl into the oven."
    ],
    "mnt/raid5/data/roboset/v0.4/heat_soup/scene_2/baking_close_oven_scene_2": [
        "Flap and close the oven."
    ],
    "home/jaydv/Documents/RoboSet/pick_banana_place_in_strainer_data": [
        "Pick banana and place it in strainer."
    ],
    "mnt/raid5/data/roboset/v0.3/pick_banana_place_in_mug_data": [
        "Pick banana and place it in mug."
    ],
    "mnt/raid5/data/roboset/v0.4/make_tea/scene_2/make_tea_pick_tea_scene_2": [
        "Pick up the tea from the container."
    ],
    "home/jaydv/Documents/RoboSet/drag_strainer_forward_data": [
        "Drag strainer forwards."
    ],
    "mnt/raid5/data/roboset/v0.3/pick_ketchup_from_strainer_place_on_table_data": [
        "Pick ketchup from strainer and place it on the table."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_4/baking_prep_place_butter_scene_4": [
        "Place the butter on the cutting board."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_1/baking_prep_slide_open_drawer_scene_1": [
        "Slide and open the drawer."
    ],
    "mnt/raid5/data/roboset/v0.3/drag_strainer_right_to_left_data": [
        "Drag strainer right to left."
    ],
    "home/jaydv/Documents/RoboSet/pick_banana_from_plate_place_on_table_data": [
        "Pick banana from plate and place on table."
    ],
    "mnt/raid5/data/roboset/v0.3/pick_ketchup_from_plate_place_on_table_data": [
        "Pick ketchup from plate and place it on table."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_1/baking_prep_slide_close_drawer_scene_1": [
        "Slide and close the drawer."
    ],
    "mnt/raid5/data/roboset/v0.3/drag_strainer_left_to_right_data": [
        "Drag strainer left to right."
    ],
    "pick_ketchup_place_on_toaster_data": [
        "Pick ketchup from table and place on toaster."
    ],
    "mnt/raid5/data/roboset/v0.4/make_tea/scene_2/make_tea_place_tea_scene_2": [
        "Place the tea into the cup."
    ],
    "mnt/raid5/data/roboset/v0.3/pick_ketchup_place_in_strainer_data": [
        "Pick ketchup from the table and place it in strainer."  
    ],
    "home/jaydv/Documents/RoboSet/pick_ketchup_place_on_plate_data": [
        "Pick ketchup from table and place on plate."
    ],
    "home/jaydv/Documents/RoboSet/drag_mug_backward_data": [
        "Drag mug backwards."
    ],
    "set_1_blocks_897": [
        "Pick up one block in the basket."
    ],
    "mnt/raid5/data/roboset/v0.4/make_tea/scene_2/make_tea_place_lid_scene_2": [
        "Place lid on the cutting board."
    ],
    "mnt/raid5/data/roboset/v0.4/heat_soup/scene_4/baking_pick_bowl_scene_4": [
        "Pick up the bowl."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_4/baking_prep_pick_butter_scene_4": [
        "Pick up the butter from the drawer."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_1/baking_prep_pick_butter_scene_1": [
        "Pick up the butter from the drawer."
    ],
    "home/jaydv/Documents/RoboSet/flap_close_toaster_oven_data": [
        "Flap close toaster."
    ],
    "home/jaydv/Documents/RoboSet/drag_mug_from_left_to_right_data": [
        "Drag mug left to right."
    ],
    "set_6_planar_push_120": [
        "Push the object from left to right."
    ],
    "mnt/raid5/data/roboset/v0.4/clean_kitchen/scene_3/clean_kitchen_slide_close_drawer_scene_3": [
        "Slide and close the drawer."
    ],
    "set_4_med_block_7": [
        "Pick up the wooden block."
    ],
    "mnt/raid5/data/roboset/v0.3/pick_banana_place_on_toaster_data": [
        "Pick banana from table and place on toaster."
    ],
    "mnt/raid5/data/roboset/v0.3/pick_ketchup_from_toaster_place_on_table_data": [
        "Pick ketchup from toaster and place it on table."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_1/baking_prep_place_butter_scene_1": [
        "Place the butter on the cutting board."
    ],
    "mnt/raid5/data/roboset/v0.4/baking_prep/scene_4/baking_prep_slide_open_drawer_scene_4": [
        "Slide and open the drawer."
    ],
    "home/jaydv/Documents/RoboSet/pick_banana_place_on_plate_data": [
        "Pick banana from table and place on plate."
    ],
    "set_8_pick_bottle_10": [
        "Pick up the bottle."
    ],
    "home/jaydv/Documents/RoboSet/pick_ketchup_place_in_toaster_data": [
        "Pick ketchup from the table and place in toaster."
    ]
}

image_shape = (240, 424, 3)
Dmanus = ['']
def stash_image_into_observation(step):
    step['observation'] = {'cam_high': [], 'cam_left_wrist': [], 'cam_right_wrist':[]}
    step['observation']['cam_high'] = step['cam_high']
    step['observation']['cam_left_wrist'] = step['cam_left_wrist']
    step['observation']['cam_right_wrist'] = step['cam_right_wrist']
    return step

def _parse_function(proto,instruction):
    # Update the keys_to_features dictionary to match the new TFRecord format
    keys_to_features = {
        'action': tf.io.FixedLenFeature([], tf.string),
        'action_gripper': tf.io.FixedLenFeature([], tf.string),
        'qpos': tf.io.FixedLenFeature([], tf.string),
        'qvel': tf.io.FixedLenFeature([], tf.string),
        'qpos_gripper': tf.io.FixedLenFeature([], tf.string),
        'qvel_gripper': tf.io.FixedLenFeature([], tf.string),
        'rgb_left': tf.io.FixedLenFeature([], tf.string),
        'rgb_right': tf.io.FixedLenFeature([], tf.string),
        'rgb_top': tf.io.FixedLenFeature([], tf.string),
        'terminate_episode': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the incoming features according to the dictionary
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Deserialize and reshape tensors as necessary
    action = tf.io.parse_tensor(parsed_features['action'], out_type=tf.float16)
    action_gripper = tf.io.parse_tensor(parsed_features['action_gripper'], out_type=tf.float16)
    qpos = tf.io.parse_tensor(parsed_features['qpos'], out_type=tf.float16)
    qvel = tf.io.parse_tensor(parsed_features['qvel'], out_type=tf.float16)
    qpos_gripper = tf.io.parse_tensor(parsed_features['qpos_gripper'], out_type=tf.float16)
    qvel_gripper = tf.io.parse_tensor(parsed_features['qvel_gripper'], out_type=tf.float16)
    rgb_left = tf.io.parse_tensor(parsed_features['rgb_left'], out_type=tf.uint8)
    rgb_right = tf.io.parse_tensor(parsed_features['rgb_right'], out_type=tf.uint8)
    rgb_top = tf.io.parse_tensor(parsed_features['rgb_top'], out_type=tf.uint8)
    terminate_episode = tf.cast(parsed_features['terminate_episode'], tf.int64)

    # Reshape or modify other fields as needed to fit the model input
    rgb_left = tf.reshape(rgb_left, image_shape)
    rgb_right = tf.reshape(rgb_right, image_shape)
    rgb_top = tf.reshape(rgb_top, image_shape)

    return {
        "action": action,
        "action_gripper": action_gripper,
        "qpos": qpos,
        "qvel": qvel,
        "qpos_gripper": qpos_gripper,
        "qvel_gripper": qvel_gripper,
        "observation": {
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_top": rgb_top
        },
        "terminate_episode": terminate_episode,
        "instruction": instruction
    }


def dataset_generator_from_tfrecords(seed):
    tfrecord_path = './data/datasets/roboset/tfrecords/'   
    failure = [f'set_{i}' for i in range(10, 18)]
    filepaths = []
    for root, dirs, files in os.walk(tfrecord_path):
        # skip datasets with failure
        fail = False
        for f in failure:
            if f in root:
                fail = True
                break
        if fail:
            continue
        
        for filename in fnmatch.filter(files, '*.tfrecord'):
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)
    
    random.seed(seed)
    random.shuffle(filepaths)
    for filepath in filepaths:
        for path in path2json:
            if path in filepath:
                instruction = path2json[path]
        raw_dataset = tf.data.TFRecordDataset(filepath)
        dataset = raw_dataset.map(lambda x: _parse_function(x,instruction))
        yield {
            'steps': dataset
        }
        
def load_dataset(seed):
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator_from_tfrecords(seed),
        output_signature={
            'steps': tf.data.DatasetSpec(
                element_spec={
                    'action': tf.TensorSpec(shape=(None), dtype=tf.float16),
                    'action_gripper': tf.TensorSpec(shape=(None), dtype=tf.float16),
                    'qpos': tf.TensorSpec(shape=(None), dtype=tf.float16),  
                    'qvel': tf.TensorSpec(shape=(None), dtype=tf.float16), 
                    'qpos_gripper': tf.TensorSpec(shape=(None), dtype=tf.float16),  
                    'qvel_gripper': tf.TensorSpec(shape=(None), dtype=tf.float16),  
                    'observation': {
                        'rgb_left': tf.TensorSpec(shape=image_shape, dtype=tf.uint8),
                        'rgb_right': tf.TensorSpec(shape=image_shape, dtype=tf.uint8),
                        'rgb_top': tf.TensorSpec(shape=image_shape, dtype=tf.uint8),
                    },
                    'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.int64),
                    'instruction': tf.TensorSpec(shape=(None), dtype=tf.string)
                }
            )
        }
    )

    return dataset


def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(tf.equal(terminate_act, tf.constant(0.0, dtype=tf.float16)),tf.constant(False),tf.constant(True))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    step['action'] = {}
    step['action']['terminate'] = step['terminate_episode']
    # undetermined action
    
    state = step['observation']
    qpos = tf.cast(step['qpos'], tf.float32)
    # qvel = tf.cast(step['qvel'], tf.float32)
    gripper_pos = tf.expand_dims(tf.cast(step['qpos_gripper'], tf.float32), axis=0)
    # delete due to all zeros
    # gripper_vel = tf.expand_dims(tf.cast(step['qvel_gripper'], tf.float32), axis=0)
    
    # state['arm_concat'] = tf.concat([qpos, qvel, gripper_pos, gripper_vel], axis=0)
    state['arm_concat'] = tf.concat([qpos, gripper_pos], axis=0)
    # state['format'] = tf.constant(
    #     "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_joint_0_pos,gripper_joint_0_vel"
    #     )
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_joint_0_pos"
        )
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
    instr = step['instruction'][0]
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    dataset = load_dataset()
    for step in dataset.take(100):
        for data in step['steps']:
            data = process_step(data)
            print(data)
            break
