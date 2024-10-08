import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_rotation_matrix, rotation_matrix_to_ortho6d

def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action

    origin_action = step['action']
    step['action']={}
    action=step['action']
    action['terminate'] = step['is_terminal']

    eef_delta_pos = origin_action[:3]

    # Ignore grip_open: -0.15~0.15

    # No base found

    # Concatenate the action
    action['arm_concat'] = eef_delta_pos

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z")

    # Convert raw state to our state
    state = step['observation']
    eef_quat = state['ee_orientation']
    eef_ang = quaternion_to_rotation_matrix(eef_quat)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    eef_ang_vel = state['ee_orientation_vel']
    eef_pos = state['ee_position']
    eef_vel = state['ee_vel']
    joint_pos = state['joint_pos']
    joint_vel = state['joint_vel']
    gripper_open = state['state'][7:8]
    
    # Concatenate the state
    state['arm_concat'] = tf.concat([joint_pos,gripper_open,joint_vel,eef_pos,eef_ang,eef_vel,eef_ang_vel],axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_open,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw")

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
    DATASET_NAME = 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
