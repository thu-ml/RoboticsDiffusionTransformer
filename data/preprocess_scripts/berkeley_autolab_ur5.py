import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, \
    quaternion_to_rotation_matrix, rotation_matrix_to_ortho6d


def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(tf.equal(terminate_act, tf.constant(0.0, dtype=tf.float32)),tf.constant(False),tf.constant(True))


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

    # Ignore action['gripper_open']: 1 if close gripper, -1 if open gripper, 0 if no change.

    # No base found

    # Concatenate the action
    arm_action = tf.concat([eef_delta_pos, eef_ang], axis=0)
    action['arm_concat'] = arm_action

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w")

    # Convert raw state to our state
    state = step['observation']
    # state['robot_state']:[joint0, joint1, joint2, joint3, joint4, joint5, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked]
    robot_state = state['robot_state']
    joint_pos=robot_state[:6]
    eef_pos = robot_state[6:9]
    eef_quat = robot_state[9:13]
    eef_ang = quaternion_to_rotation_matrix(eef_quat)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    # gripper_is_closed is binary: 0 = fully open; 1 = fully closed
    grip_closed = robot_state[13:14]
    grip_open= 1-grip_closed
    # action_blocked is binary: 0 = not blocked; 1 = blocked
    # action_blocked = robot_state[14:15]
    
    # Concatenate the state
    state['arm_concat'] = tf.concat([joint_pos, grip_open,eef_pos,eef_ang], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
    DATASET_NAME = 'berkeley_autolab_ur5'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
