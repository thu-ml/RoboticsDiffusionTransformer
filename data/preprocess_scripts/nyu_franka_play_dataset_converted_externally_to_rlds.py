import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, euler_to_rotation_matrix, \
    rotation_matrix_to_ortho6d

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

    origin_action = step['action']
    step['action']={}
    action=step['action']
    action['terminate']=terminate_act_to_bool(origin_action[14])
    joint_vel=origin_action[:7]
    # gripper_open: 1-open, -1-closed
    
    grip_open=tf.where(tf.equal(origin_action[13:14],tf.constant(1.0)),tf.constant(1.0),tf.constant(0.0))
    eef_pos=origin_action[7:10]
    eef_ang=origin_action[10:13]
    eef_ang = euler_to_quaternion(eef_ang)
    # No base found

    # Concatenate the action
    action['arm_concat'] = tf.concat([joint_vel,grip_open,eef_pos,eef_ang],axis=0)

    # Write the action format
    action['format'] = tf.constant(
        "arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_joint_0_pos,eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w")

    # Convert raw state to our state
    state = step['observation']
    # Concatenate the state
    state_arm_joint = state['state'][:7]
    eef_pos = state['state'][7:10]
    eef_ang = euler_to_rotation_matrix(state['state'][10:13])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    state['arm_concat'] = tf.concat([state_arm_joint,eef_pos,eef_ang],axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
    DATASET_NAME = 'nyu_franka_play_dataset_converted_externally_to_rlds'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
