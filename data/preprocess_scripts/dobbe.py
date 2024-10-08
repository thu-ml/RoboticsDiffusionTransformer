import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_rotation_matrix, rotation_matrix_to_ortho6d


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    arm_action = step['action']

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    action['arm_concat'] = arm_action
    # Write the action format
    action['format'] = tf.constant(
        "eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw,gripper_open")
    action['terminate'] = step['is_terminal']

    # The state has any problem?
    state = step['observation']
    eef_pos = state['xyz']
    # Clip eef_pos to be [-10, 10] for stability
    eef_pos = tf.clip_by_value(eef_pos, -10, 10)
    eef_ang = state['rot']
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    grip_pos = state['gripper']

    # Concatenate the state
    state['arm_concat'] = tf.concat([
        grip_pos,eef_pos,eef_ang], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
    from tqdm import tqdm
    import numpy as np

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'dobbe'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')
    # dataset = dataset.filter(
    #     lambda x: tf.math.less(
    #         tf.math.reduce_max(tf.math.abs(
    #             tf.convert_to_tensor(
    #                 [step['observation']['xyz'] for step in x['steps']]))), 3))

    # Inspect the dataset
    for i, episode in tqdm(enumerate(dataset), total=5208):
        res = []
        for step in episode['steps']:
            res.append(step['observation']['xyz'].numpy())
        max_val = np.max(np.abs(res))
        if max_val > 2:
            print(f"Episode {i} has a max value of {max_val}")
