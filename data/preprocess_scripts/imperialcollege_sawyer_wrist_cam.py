import tensorflow as tf

from data.utils import clean_task_instruction


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action

    origin_action = step['action']
    step['action'] = {}
    action = step['action']

    # Multiplied by 10 Hz to get units m/s and rad/s
    eef_delta_pos = origin_action[:3] * 10
    # delta ZYX euler angles == roll/pitch/yaw velocities
    eef_ang = origin_action[3:6] * 10
    grip_open = 1 - origin_action[6:7]

    # No base found

    # Concatenate the action
    action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang, grip_open],axis=0)

    # Write the action format
    action['format'] = tf.constant(
        "eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw,gripper_open")

    # Convert raw state to our state
    # state = step['observation']
    # # Concatenate the state
    # grip_open=state['state']
    # state['arm_concat'] = grip_open

    # Write the state format
    # state['format'] = tf.constant(
    #     "gripper_open")

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
    DATASET_NAME = 'imperialcollege_sawyer_wrist_cam'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
