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
    # 	Robot action, consists of [7 delta joint pos,1x gripper binary state].
    delta_joint_pos = action[:7]
    grip_open = tf.expand_dims(1 - action[7], axis=0)

    # Concatenate the action
    # action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
    step['action'] = {}
    action = step['action']
    action['arm_concat'] = tf.concat([delta_joint_pos, grip_open], axis=0)
    action['terminate'] = step['is_terminal']

    # Write the action format
    action['format'] = tf.constant(
        "arm_joint_0_delta_pos,arm_joint_1_delta_pos,arm_joint_2_delta_pos,arm_joint_3_delta_pos,arm_joint_4_delta_pos,arm_joint_5_delta_pos,arm_joint_6_delta_pos,gripper_open")

    # Convert raw state to our state
    state = step['observation']
    # xArm joint positions (7 DoF).
    arm_joint_pos = state['joint_pos']
    # Binary gripper state (1 - closed, 0 - open)
    grip_open = tf.expand_dims(1 - tf.cast(state['gripper'],dtype=tf.float32), axis=0)
    # Gripper pose, robot frame, [3 position, 4 rotation]
    gripper_pose = state['pose'][:3]
    # gripper_ang = quaternion_to_euler(state['pose'][3:7])
    gripper_ang = quaternion_to_rotation_matrix(state['pose'][3:7])
    gripper_ang = rotation_matrix_to_ortho6d(gripper_ang)

    # Concatenate the state
    state['arm_concat'] = tf.concat([arm_joint_pos, gripper_pose,gripper_ang, grip_open], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,gripper_open")

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
