import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_rotation_matrix, rotation_matrix_to_ortho6d, \
    quaternion_to_rotation_matrix


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    action = step['action']

    # Robot action
    eef_pos = action[:3]
    eef_ang = action[3:6]
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    gripper_open = action[6:7]

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    
    arm_action = tf.concat([eef_pos, eef_ang, gripper_open], axis=0)
    action['arm_concat'] = arm_action
    action['terminate'] = step['is_terminal']

    # Write the action format
    action['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,gripper_open")

    # Convert raw state to our state
    # Robot state
    state = step['observation']
    eef_pos = state['eef_pose'][:3]
    eef_ang = state['eef_pose'][3:]
    eef_ang = quaternion_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    eef_pos_vel = state['eef_vel'][:3]
    eef_ang_vel = state['eef_vel'][3:]
    joint_pos = state['joint_pos']
    joint_vel = state['joint_vel']
    grip_pos = 1 - state['state_gripper_pose']
    grip_pos = tf.expand_dims(grip_pos, axis=0)

    # Concatenate the state
    state['arm_concat'] = tf.concat([
        joint_pos,joint_vel,grip_pos,eef_pos,eef_ang,eef_pos_vel,eef_ang_vel], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw")

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
        # Refine language instruction:
        'object': 'brick and insert it into the slot of the matching shape'    
    }
    instr = step['language_instruction']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    pass
