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
    # Robot action, consists of [3x end-effector position residual, 3x end-effector axis-angle residual, 
    # 7x robot joint k_p gain coefficient, 7x robot joint damping ratio coefficient].
    # The action residuals are global, i.e. multiplied on theleft-hand side of the current end-effector state.
    eef_delta_pos = action[:3]
    eef_ang = action[3:6] 
    eef_ang = euler_to_quaternion(eef_ang)

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    action['terminate'] = step['is_terminal']
    action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang,], axis=0)

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w")

    # Convert raw state to our state
    state = step['observation']
    # Robot state, consists of [joint_states, end_effector_pose].Joint states are 14-dimensional, formatted in the order of [q_0, w_0, q_1, w_0, ...].
    # In other words, joint positions and velocities are interleaved.The end-effector pose is 7-dimensional, formatted in the order of [position, quaternion].The quaternion is formatted in (x,y,z,w) order. The end-effector pose references the tool frame, in the center of the two fingers of the gripper.
    joint_states = state['state'][:14]
    arm_joint_pos = joint_states[::2]
    arm_joint_vel = joint_states[1::2]
    eef_pos = state['state'][14:17]
    # eef_ang = quaternion_to_euler(state['state'][17:21])
    eef_ang = quaternion_to_rotation_matrix(state['state'][17:21])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    # Concatenate the state
    state['arm_concat'] = tf.concat([arm_joint_pos, arm_joint_vel, eef_pos, eef_ang], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
