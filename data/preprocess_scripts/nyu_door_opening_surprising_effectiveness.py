import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion


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

    # Multiplied by 3 Hz to get units m/s and rad/s
    eef_delta_pos = action['world_vector'] * 3
    eef_ang = action['rotation_delta'] * 3
    # Origin: [-0.28, 0.96]: open, close
    # 1-Origin: [0.04, 1.28]: close, open
    grip_open = 1 - action['gripper_closedness_action']
    # base_delta_pos = action['base_displacement_vector']
    # base_delta_ang = action['base_displacement_vertical_rotation']

    # Concatenate the action
    arm_action = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
    action['arm_concat'] = arm_action
    # base_action = tf.concat([base_delta_pos, base_delta_ang], axis=0)
    # action['base_concat'] = base_action

    # Write the action format
    action['format'] = tf.constant(
        "eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw,gripper_open")

    # Convert raw state to our state
    # state = step['observation']
    # eef_pos = state['base_pose_tool_reached'][:3]
    # eef_ang = quaternion_to_euler(state['base_pose_tool_reached'][3:])
    # grip_open = 1 - state['gripper_closed']

    # create empty tensor
    # state['arm_concat'] = tf.constant([0, 0, 0, 0, 0, 0], dtype=tf.float32)

    # Write the state format
    # state['format'] = tf.constant(
    #     "")

    # Define the task instruction
    step['observation']['natural_language_instruction'] = tf.constant(
        "Open the cabinet door.")

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'nyu_door_opening_surprising_effectiveness'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset.take(1):
        for step in episode['steps']:
            print(step)
