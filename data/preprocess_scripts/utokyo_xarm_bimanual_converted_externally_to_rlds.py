import json
import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_rotation_matrix, rotation_matrix_to_ortho6d


def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.reduce_all(tf.equal(terminate_act, tf.constant([1, 0, 0], dtype=tf.int32)))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    action = step['action']
    
    # TODO 
    # action : Tensor	(14,)	[3x EEF position (L), 3x EEF orientation yaw/pitch/roll (L), 1x gripper open/close position (L), 3x EEF position (R), 3x EEF orientation yaw/pitch/roll (R), 1x gripper open/close position (R)].

    eef_pos_left = action[0:3]
    eef_angle_left = tf.gather(action[3:6], [2, 1, 0])
    eef_angle_left = euler_to_rotation_matrix(eef_angle_left)
    eef_angle_left = rotation_matrix_to_ortho6d(eef_angle_left)
    gripper_open_left = 1 - action[6:7]
    eef_pos_right = action[7:10]
    eef_angle_right = tf.gather(action[10:13], [2, 1, 0])
    eef_angle_right = euler_to_rotation_matrix(eef_angle_right)
    eef_angle_right = rotation_matrix_to_ortho6d(eef_angle_right)
    gripper_open_right = 1 - action[13:14]

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    
    # Concatenate the action
    arm_action = tf.concat([eef_pos_left,eef_angle_left,gripper_open_left,eef_pos_right,eef_angle_right,gripper_open_right], axis=0)
    action['arm_concat'] = arm_action
    action['terminate'] = step['is_terminal']
    
    # print("action len:", len(action['arm_concat']) + len(action['base_concat']))

    action['format'] = tf.constant(
        "left_eef_pos_x,left_eef_pos_y,left_eef_pos_z,left_eef_angle_0,left_eef_angle_1,left_eef_angle_2,left_eef_angle_3,left_eef_angle_4,left_eef_angle_5,left_gripper_open,right_eef_pos_x,right_eef_pos_y,right_eef_pos_z,right_eef_angle_0,right_eef_angle_1,right_eef_angle_2,right_eef_angle_3,right_eef_angle_4,right_eef_angle_5,right_gripper_open")

    # action good for kuka same as example
    
    # Convert raw state to our state
    action = step['observation']['action_l']
    # [3x EEF position, 3x EEF orientation yaw/pitch/roll, 1x gripper open/close position].
    eef_pos_left = action[0:3]
    eef_angle_left = tf.gather(action[3:6], [2, 1, 0])
    eef_angle_left = euler_to_rotation_matrix(eef_angle_left)
    eef_angle_left = rotation_matrix_to_ortho6d(eef_angle_left)
    gripper_open_left = 1 - action[6:7]

    action = step['observation']['action_r']
    eef_pos_right = action[0:3]
    eef_angle_right = tf.gather(action[3:6], [2, 1, 0])
    eef_angle_right = euler_to_rotation_matrix(eef_angle_right)
    eef_angle_right = rotation_matrix_to_ortho6d(eef_angle_right)
    gripper_open_right = 1 - action[6:7]

    # Write the state format TODO how to link 12 joint pos to 7 joint pos ??
    state = step['observation']
    # Concatenate the state
    state['arm_concat'] = tf.concat([eef_pos_left,eef_angle_left,gripper_open_left,eef_pos_right,eef_angle_right,gripper_open_right], axis=0)
    state['format'] = tf.constant(
        "left_eef_pos_x,left_eef_pos_y,left_eef_pos_z,left_eef_angle_0,left_eef_angle_1,left_eef_angle_2,left_eef_angle_3,left_eef_angle_4,left_eef_angle_5,left_gripper_open,right_eef_pos_x,right_eef_pos_y,right_eef_pos_z,right_eef_angle_0,right_eef_angle_1,right_eef_angle_2,right_eef_angle_3,right_eef_angle_4,right_eef_angle_5,right_gripper_open")

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
    instr = step['language_instruction']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'utokyo_xarm_bimanual_converted_externally_to_rlds'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # with open('example.txt', 'w') as file:
    # Inspect the dataset
    
    episode_num = len(dataset)
    print(f"episode_num: {episode_num}")
    for episode in dataset.take(1):
        # print("episode")
        # print(list(episode.keys()))
        for step in episode['steps']:
            process_step(step)
            break
