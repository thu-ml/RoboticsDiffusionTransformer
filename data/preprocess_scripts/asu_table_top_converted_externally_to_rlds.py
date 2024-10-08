import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_euler


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
    # Robot action, consists of [7x joint velocities, 2x gripper velocities, 1x terminate episode].
    # NOTE the dimension of action is actually 7, so only 7x joint velocities exists
    joint_vel = action[:7]
    gripper_vel = action[7:9]
    # there are extra action_delta information
    # Robot delta action, consists of [7x joint velocities, 2x gripper velocities, 1x terminate episode].
    # action_delta = step['action_delta']
    
    # Concatenate the action
    step['action'] = {}
    action = step['action']
    action['arm_concat'] = tf.concat([joint_vel, gripper_vel], axis=0)
    action['terminate'] = step['is_terminal']

    # Write the action format
    action['format'] = tf.constant(
        "arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_joint_0_vel,gripper_joint_1_vel")
    
    # Convert raw state to our state
    state = step['observation']
    state_vec = state['state']
    # Robot state, consists of [6x robot joint angles, 1x gripper position].
    arm_joint_ang = state_vec[:6]
    grip_pos = state_vec[6:7]
    # Robot joint velocity, consists of [6x robot joint angles, 1x gripper position].
    state_vel = state['state_vel']
    arm_joint_vel = state_vel[:6]
    grip_vel = state_vel[6:7]

    # Concatenate the state
    state['arm_concat'] = tf.concat([arm_joint_ang,arm_joint_vel,grip_pos,grip_vel], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,gripper_joint_0_pos,gripper_joint_0_vel")

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
