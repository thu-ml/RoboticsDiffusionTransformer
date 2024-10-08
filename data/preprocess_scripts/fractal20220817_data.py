import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, quaternion_to_rotation_matrix,\
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
    eef_ang = action['rotation_delta']
    eef_ang = euler_to_quaternion(eef_ang)
    grip_open = 1 - (action['gripper_closedness_action'] + 1) / 2
    # Multiplied by 3 Hz to get units m/s and rad/s
    base_delta_pos = action['base_displacement_vector'] * 3
    base_delta_ang = action['base_displacement_vertical_rotation'] * 3

    # Concatenate the action
    arm_action = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
    action['arm_concat'] = arm_action
    base_action = tf.concat([base_delta_pos, base_delta_ang], axis=0)
    action['base_concat'] = base_action

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open,base_vel_x,base_vel_y,base_angular_vel")

    # Convert raw state to our state
    state = step['observation']
    eef_pos = state['base_pose_tool_reached'][:3]
    # eef_ang = quaternion_to_euler(state['base_pose_tool_reached'][3:])
    eef_ang = quaternion_to_rotation_matrix(state['base_pose_tool_reached'][3:])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    grip_open = 1 - state['gripper_closed']
    
    # Concatenate the state
    state['arm_concat'] = tf.concat([eef_pos, eef_ang, grip_open], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,gripper_open")

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
    DATASET_NAME = 'fractal20220817_data'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
