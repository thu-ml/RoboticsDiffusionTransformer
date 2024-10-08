import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, euler_to_rotation_matrix,\
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
    # Robot action, consists of [3x robot EEF position, 3x robot EEF orientation yaw/pitch/roll calculated with scipy Rotation.as_euler(="zxy") Class].
    eef_delta_pos = action[:3]
    eef_ang = tf.gather(action[3:6], [1, 2, 0])  # To Rotation.as_euler("xyz")
    eef_ang = euler_to_quaternion(eef_ang)
    # After watching the episode, I found that the last action is the gripper open/close action, where 0 means open and 1 means close.
    grip_open = tf.expand_dims(1 - action[6], axis=0)
    # Concatenate the action
    step['action'] = {}
    action = step['action']
    action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang,grip_open], axis=0)
    action['terminate'] = step['is_terminal']
    
    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")

    # Convert raw state to our state
    state = step['observation']
    state_vec = state['state']
    # Robot state, consists of [3x robot EEF position, 3x robot EEF orientation yaw/pitch/roll calculated with scipy Rotation.as_euler(="zxy") Class].
    eef_pos = state_vec[:3]
    eef_ang = tf.gather(state_vec[3:6], [1, 2, 0])  # To Rotation.as_euler("xyz")
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)

    # Concatenate the state
    state['arm_concat'] = tf.concat([ eef_pos, eef_ang], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
