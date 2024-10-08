import tensorflow as tf

from data.utils import clean_task_instruction,quaternion_to_euler

def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action

    origin_action = step['action']
    step['action']={}
    action=step['action']

    eef_delta_pos = origin_action[:3]
    # delta ZYX euler angles
    # eef_ang=quaternion_to_euler(origin_action[3:7])
    eef_ang = origin_action[3:7]
    grip_open=origin_action[7:8]

    # No base found

    # Concatenate the action
    action['arm_concat'] = tf.concat([eef_delta_pos,eef_ang,grip_open],axis=0)

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")

    # Convert raw state to our state
    state = step['observation']
    # Concatenate the state
    # 7x robot joint angles, 1x gripper status, 6x joint torques, 6x end-effector force
    arm_joint_ang=state['state'][:7]

    grip_open=state['state'][7:8]

    state['arm_concat'] = tf.concat([arm_joint_ang,grip_open],axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_open")

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
    DATASET_NAME = 'iamlab_cmu_pickup_insert_converted_externally_to_rlds'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
