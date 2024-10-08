import tensorflow as tf
import tensorflow_datasets as tfds
from data.utils import clean_task_instruction, quaternion_to_euler


def load_dataset():
    builder = tfds.builder('robomimic_ph/lift_ph_image')
    builder.download_and_prepare()
    ds = builder.as_dataset(split='train', shuffle_files=True)
    return ds

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
    # format refers to https://www.tensorflow.org/datasets/catalog/robomimic_mg
    # Convert raw action to our action
    eef = step['action']
    step['action'] = {}
    action = step['action']
    action['terminate'] = step['is_terminal']

    eef_delta_pos = eef[:3]
    eef_ang = quaternion_to_euler(eef[3:])
    
    # No base found

    # Concatenate the action
    arm_action = tf.concat([eef_delta_pos, eef_ang], axis=0)
    action['arm_concat'] = arm_action

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_roll,eef_delta_angle_pitch,eef_delta_angle_yaw")

    # Convert raw state to our state
    state = step['observation']
    arm_joint_pos = state['robot0_joint_pos']
    arm_joint_vel = state['robot0_joint_vel']
    gripper_pos = state['robot0_gripper_qpos']
    gripper_vel = state['robot0_gripper_qvel']
    eef_pos = state['robot0_eef_pos']
    eef_ang = quaternion_to_euler(state['robot0_eef_quat'])
    
    state['arm_concat'] = tf.concat([arm_joint_pos, arm_joint_vel, gripper_pos,gripper_vel,eef_pos,eef_ang], axis=0)
    # convert to tf32
    state['arm_concat'] = tf.cast(state['arm_concat'], tf.float32)
    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_joint_0_pos,gripper_joint_1_pos,gripper_joint_0_vel,gripper_joint_1_vel,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_roll,eef_angle_pitch,eef_angle_yaw")

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
    # manual added by lbg
    instr = "lift the object on the table"
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'roboturk'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all').take(1)

    # Inspect the dataset
    ze=tf.constant(0.0)
    for episode in dataset:
        for step in episode['steps']:
            print(step)
            break
