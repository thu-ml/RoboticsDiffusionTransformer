import tensorflow as tf
import tensorflow_datasets as tfds
from data.utils import clean_task_instruction, quaternion_to_euler


def load_dataset():
    builder = tfds.builder('robomimic_ph/transport_ph_image')
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

    left_eef_delta_pos = eef[:3]
    left_eef_ang = quaternion_to_euler(eef[3:7])
    
    right_eef_delta_pos = eef[7:10]
    right_eef_ang = quaternion_to_euler(eef[10:])
    
    # No base found

    # Concatenate the action
    arm_action = tf.concat([left_eef_delta_pos,left_eef_ang,right_eef_delta_pos,right_eef_ang], axis=0)
    action['arm_concat'] = arm_action

    # Write the action format
    action['format'] = tf.constant(
        "left_eef_delta_pos_x,left_eef_delta_pos_y,left_eef_delta_pos_z,left_eef_delta_angle_roll,left_eef_delta_angle_pitch,left_eef_delta_angle_yaw,right_eef_delta_pos_x,right_eef_delta_pos_y,right_eef_delta_pos_z,right_eef_delta_angle_roll,right_eef_delta_angle_pitch,right_eef_delta_angle_yaw")

    # Convert raw state to our state
    state = step['observation']
    left_arm_joint_pos = state['robot0_joint_pos']
    left_arm_joint_vel = state['robot0_joint_vel']
    left_gripper_pos = state['robot0_gripper_qpos']
    left_gripper_vel = state['robot0_gripper_qvel']
    left_eef_pos = state['robot0_eef_pos']
    left_eef_ang = quaternion_to_euler(state['robot0_eef_quat'])
    
    right_arm_joint_pos = state['robot1_joint_pos']
    right_arm_joint_vel = state['robot1_joint_vel']
    right_gripper_pos = state['robot1_gripper_qpos']
    right_gripper_vel = state['robot1_gripper_qvel']
    right_eef_pos = state['robot1_eef_pos']
    right_eef_ang = quaternion_to_euler(state['robot1_eef_quat'])
    
    arm_joint_pos = tf.concat([left_arm_joint_pos, right_arm_joint_pos], axis=0)
    arm_joint_vel = tf.concat([left_arm_joint_vel, right_arm_joint_vel], axis=0)
    gripper_pos = tf.concat([left_gripper_pos, right_gripper_pos], axis=0)
    gripper_vel = tf.concat([left_gripper_vel, right_gripper_vel], axis=0)
    eef_pos = tf.concat([left_eef_pos, right_eef_pos], axis=0)
    eef_ang = tf.concat([left_eef_ang, right_eef_ang], axis=0)
    
    state['arm_concat'] = tf.concat([arm_joint_pos, arm_joint_vel, gripper_pos,gripper_vel,eef_pos,eef_ang], axis=0)
    # convert to tf32
    state['arm_concat'] = tf.cast(state['arm_concat'], tf.float32)
    # Write the state format
    state['format'] = tf.constant(
        "left_arm_joint_0_pos,left_arm_joint_1_pos,left_arm_joint_2_pos,left_arm_joint_3_pos,left_arm_joint_4_pos,left_arm_joint_5_pos,left_arm_joint_6_pos,right_arm_joint_0_pos,right_arm_joint_1_pos,right_arm_joint_2_pos,right_arm_joint_3_pos,right_arm_joint_4_pos,right_arm_joint_5_pos,right_arm_joint_6_pos,left_gripper_joint_0_pos,left_gripper_joint_1_pos,right_gripper_joint_0_pos,right_gripper_joint_1_pos,left_arm_joint_0_vel,left_arm_joint_1_vel,left_arm_joint_2_vel,left_arm_joint_3_vel,left_arm_joint_4_vel,left_arm_joint_5_vel,left_arm_joint_6_vel,right_arm_joint_0_vel,right_arm_joint_1_vel,right_arm_joint_2_vel,right_arm_joint_3_vel,right_arm_joint_4_vel,right_arm_joint_5_vel,right_arm_joint_6_vel,left_gripper_joint_0_vel,left_gripper_joint_1_vel,right_gripper_joint_0_vel,right_gripper_joint_1_vel,left_eef_pos_x,left_eef_pos_y,left_eef_pos_z,left_eef_angle_roll,left_eef_angle_pitch,left_eef_angle_yaw,right_eef_pos_x,right_eef_pos_y,right_eef_pos_z,right_eef_angle_roll,right_eef_angle_pitch,right_eef_angle_yaw")

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
    instr = "transport the object from left hand to right hand"
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
