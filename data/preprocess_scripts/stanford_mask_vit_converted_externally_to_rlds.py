import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, euler_to_rotation_matrix, \
    rotation_matrix_to_ortho6d


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
    # Robot action, consists of [3x change in end effector position, 1x gripper yaw, 1x open/close gripper (-1 means to open the gripper, 1 means close)].
    eef_delta_pos = action[:3]
    eef_yaw = action[3:4]
    eef_ang = tf.stack([0,0,eef_yaw[0]],axis=0)
    eef_ang = euler_to_quaternion(eef_ang)
    grip_open = tf.expand_dims((1 - action[4]) / 2, axis=0)

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
    
    action['terminate'] = step['is_terminal']
    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")


    # Convert raw state to our state
    state = step['observation']['state']
    # Robot state, consists of [7x robot joint angles, 7x robot joint velocities,1x gripper position].
    arm_joint_pos = state[:7]
    arm_joint_vel = state[7:14]
    gripper_pos = state[14:15]
    # Robot end effector pose, consists of [3x Cartesian position, 1x gripper yaw, 1x gripper position]. This is the state used in the MaskViT paper.
    eef = step['observation']['end_effector_pose']
    eef_pos = eef[:3]
    gripper_yaw = eef[3:4]
    eef_ang = tf.stack([0.0,0.0,gripper_yaw[0]],axis=0)
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    # gripper_pos = eef[4:5]

    # Concatenate the state
    state = step['observation']
    state['arm_concat'] = tf.concat([arm_joint_pos,arm_joint_vel,gripper_pos,eef_pos,eef_ang], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_joint_0_pos,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
