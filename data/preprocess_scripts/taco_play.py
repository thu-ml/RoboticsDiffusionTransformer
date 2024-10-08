import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, euler_to_rotation_matrix, \
    rotation_matrix_to_ortho6d


def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.equal(terminate_act, tf.constant(1.0, dtype=tf.float32))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    action = step['action']
    action['terminate'] = terminate_act_to_bool(action['terminate_episode'])

    # action['actions']: absolute desired values for gripper pose
    the_actions=action['actions']
    # First 6 dimensions are x, y, z, yaw, pitch, roll
    eef_pos=the_actions[:3]
    eef_ang = tf.concat([the_actions[5:6],the_actions[4:5],the_actions[3:4]],axis=0)
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    open_gripper=the_actions[6:]
    # Last dimension is open_gripper (-1 is open gripper, 1 is close)
    grip_open=tf.reshape(tf.where(open_gripper<0,1.0,0.0),(1,)) 
    # -1 -> 1.0, 1->0.0

    # No base found

    # Concatenate the action
    arm_action = tf.concat([eef_pos, eef_ang, grip_open], axis=0)
    action['arm_concat'] = arm_action

    # Write the action format
    action['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,gripper_open")

    # Convert raw state to our state
    state = step['observation']
    robot_obs=state['robot_obs']
    # robot_obs: EE position (3), EE orientation in euler angles (3), gripper width (1), joint positions (7), gripper action (1)
    eef_pos=robot_obs[:3]
    eef_ang=robot_obs[3:6]
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    gripper_open=robot_obs[6:7] * 12.3843 # rescale to [0, 1]
    joint_pos=robot_obs[7:14]
    
    # Concatenate the state
    state['arm_concat'] = tf.concat([joint_pos,gripper_open,eef_pos,eef_ang],axis=0)

    # Write the state format
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
    instr = step['observation']['natural_language_instruction']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'taco_play'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
