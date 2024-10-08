import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_rotation_matrix, \
    rotation_matrix_to_ortho6d

def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action

    origin_action = step['action']
    step['action']={}
    action=step['action']

    # No base found
    eef_pos_delta=origin_action[:3]
    eef_quat_delta=origin_action[3:7]
    eef_ang = eef_quat_delta
    # eef_ang=quaternion_to_euler(eef_quat_delta)
    grip_open=-origin_action[7:8]

    action["arm_concat"]=tf.concat([eef_pos_delta,eef_ang,grip_open],axis=0)
    action['format']=tf.constant("eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")
    # Ignore action
    action['terminate'] = step['is_terminal']

    # Convert raw state to our state
    state = step['observation']
    # Concatenate the state
    
    ''' 'state':
    {
      'ee_pos': EEF position (3,)
      'ee_quat': EEF orientation (4,)
      'ee_pos_vel': EEF linear velocity (3,)
      'ee_ori_vel': EEF angular velocity (3,)
      'joint_positions': Joint positions (7,)
      'joint_velocities': Joint velocities (7,)
      'joint_torques': Joint torques (7,)
      'gripper_width': Gripper width (1,)
    }'''
    eef_pos=state['state'][:3]
    # eef_ang=quaternion_to_euler(state['state'][3:7])
    eef_ang = quaternion_to_rotation_matrix(state['state'][3:7])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    eef_pos_vel=state['state'][7:10]
    eef_ang_vel=state['state'][10:13]
    arm_joint_pos=state['state'][13:20]
    arm_joint_vel=state['state'][20:27]
    arm_joint_tor=state['state'][27:34]
    # gripper_width?
    grip_open=state['state'][34:35] * 12.507    # rescale to [0, 1]

    state['arm_concat'] = tf.concat([arm_joint_pos,grip_open,arm_joint_vel,\
    eef_pos,eef_ang,eef_pos_vel,eef_ang_vel],axis=0)

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
    DATASET_NAME = 'furniture_bench_dataset_converted_externally_to_rlds'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
