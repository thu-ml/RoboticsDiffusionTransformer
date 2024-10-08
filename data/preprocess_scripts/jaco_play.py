import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_rotation_matrix, \
    rotation_matrix_to_ortho6d


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
    action['terminate'] = terminate_act_to_bool(action['terminate_episode'])
    eef_delta_pos = action['world_vector']
    # eef_ang = action['rotation_delta']
    # (NOTE) due to the formality problem, grip_open is not used
    # grip_open = 1 - (action['gripper_closedness_action'] ) / 2
    # base_delta_pos = action['base_displacement_vector']
    # base_delta_ang = action['base_displacement_vertical_rotation']

    # Concatenate the action
    arm_action = eef_delta_pos
    action['arm_concat'] = arm_action
    # base_action = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    # action['base_concat'] = None

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z")

    # Convert raw state to our state
    state = step['observation']
    joint_pos = state['joint_pos']
    eef_pos = state['end_effector_cartesian_pos'][:3]
    eef_quat = state['end_effector_cartesian_pos'][3:]
    eef_ang = quaternion_to_rotation_matrix(eef_quat)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    eef_vel = state['end_effector_cartesian_velocity'][:3]
    # We do not use angular velocity since it is very inaccurate in this environment
    # eef_angular_vel = state['end_effector_cartesian_velocity'][3:]
    # Concatenate the state
    state['arm_concat'] = tf.concat([joint_pos, eef_pos, eef_ang, eef_vel], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,gripper_joint_0_pos,gripper_joint_1_pos,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,eef_vel_x,eef_vel_y,eef_vel_z")

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
    instr = step['observation']['natural_language_instruction']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'jaco_play'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
