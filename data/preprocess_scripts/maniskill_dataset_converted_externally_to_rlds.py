import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_euler, euler_to_quaternion


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

    # Robot action, consists of [3x end effector delta target position, 3x end effector delta target orientation in axis-angle format, 1x gripper target position (mimic for two fingers)]. 
    # For delta target position, an action of -1 maps to a robot movement of -0.1m, and action of 1 maps to a movement of 0.1m.
    # For delta target orientation, its encoded angle is mapped to a range of [-0.1rad, 0.1rad] for robot execution. For example, an action of [1, 0, 0] means rotating along the x-axis by 0.1 rad. 
    # For gripper target position, an action of -1 means close, and an action of 1 means open.
    eef_delta_pos = action[:3] * 0.1
    eef_ang = action[3:6] * 0.1
    eef_ang = euler_to_quaternion(eef_ang)
    grip_open = tf.expand_dims((action[6] + 1) / 2, axis=0)

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    
    arm_action = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
    action['arm_concat'] = arm_action
    action['terminate'] = step['is_terminal']

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")

    # Convert raw state to our state
    # Robot state, consists of [7x robot joint angles, 2x gripper position, 7x robot joint angle velocity, 2x gripper velocity]. Angle in radians, position in meters.
    state = step['observation']['state']
    arm_joint_pos = state[:7]
    gripper_pos = state[7:9] * 25   # rescale to [0, 1]
    # We do not use velocity since it is very inaccurate in this environment
    # arm_joint_vel = state[9:16]
    # gripper_vel = state[16:18]

    # Concatenate the state
    
    state = step['observation']
    state['arm_concat'] = tf.concat([
        arm_joint_pos, gripper_pos], axis=0)
    
    # Robot base pose in the world frame, consists of [x, y, z, qw, qx, qy, qz]. 
    # The first three dimensions represent xyz positions in meters. The last four dimensions are the quaternion representation of rotation
    # base_pose = step['observation']['base_pose']
    # base_pose_xyz = base_pose[:3]
    # base_pose_ang = quaternion_to_euler(base_pose[3:])
    # processed_base_pose = tf.concat([base_pose_xyz, base_pose_ang], axis=0)
    
    # state['arm_concat'] = tf.concat([state['arm_concat'], processed_base_pose], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_joint_0_pos,gripper_joint_1_pos")

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
    DATASET_NAME = 'maniskill_dataset_converted_externally_to_rlds'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset.take(1):
        for step in episode['steps']:
            print(step['is_last'])
