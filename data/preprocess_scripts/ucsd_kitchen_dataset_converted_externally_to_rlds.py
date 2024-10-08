import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_euler, euler_to_quaternion


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
    
    # 8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination action.
    action = step['action']
    eef_delta_pos = action[:3]
    eef_ang = action[3:6]
    eef_ang = euler_to_quaternion(eef_ang)
    grip_open = tf.expand_dims(1 - action[6], axis=0)

    # Concatenate the action
    # action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
    step['action'] = {}
    action = step['action']
    
    action['terminate'] = step['is_terminal']
    action['arm_concat'] = tf.concat([eef_delta_pos, eef_ang, grip_open], axis=0)
   
    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")

    # Convert raw state to our state
    state = step['observation']['state']
    # 21-dimensional joint states, consists of robot joint angles, joint velocity and joint torque.
    arm_joint_pos = state[:7]
    arm_joint_vel = state[7:14]
    # arm_joint_torque = state[14:21]

    # Concatenate the state
    
    state = step['observation']
    state['arm_concat'] = tf.concat([arm_joint_pos, arm_joint_vel], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel")

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
