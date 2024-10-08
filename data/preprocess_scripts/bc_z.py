import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, \
    euler_to_rotation_matrix, rotation_matrix_to_ortho6d


def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.reduce_all(tf.equal(terminate_act, tf.constant([1, 0, 0], dtype=tf.float32)))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    
    # Convert raw action to our action
    action = step['action']
    # The next 10 actions for the positions. Each action is a 3D delta to add to current position.
    eef_delta_pos = action['future/xyz_residual'][:3]
    # The next 10 actions for the rotation. Each action is a 3D delta to add to the current axis angle.
    eef_ang = action['future/axis_angle_residual'][:3]
    eef_ang = euler_to_quaternion(eef_ang)
    # The next 10 actions for the gripper. Each action is the value the gripper closure should be changed to (notably it is not a delta.)
    grip_open = tf.cast(tf.expand_dims(1 - action['future/target_close'][0], axis=0), dtype=tf.float32)

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    
    action['terminate'] = step['is_terminal']
    action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
   
    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")

    # Convert raw state to our state
    state = step['observation']

    gripper_ang = state['present/axis_angle']
    gripper_ang = euler_to_rotation_matrix(gripper_ang)
    gripper_ang = rotation_matrix_to_ortho6d(gripper_ang)
    gripper_pos = state['present/xyz']
    # How much the gripper is currently closed. Scaled from 0 to 1, but not all values from 0 to 1 are reachable. The range in the data is about 0.2 to 1
    gripper_open = 1- state['present/sensed_close']


    # Concatenate the state
    state = step['observation']
    state['arm_concat'] = tf.concat([gripper_pos, gripper_ang, gripper_open], axis=0)

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
