import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_rotation_matrix, rotation_matrix_to_ortho6d


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    action_dict = step['action_dict']

    # Robot action
    eef_pos = action_dict['cartesian_position'][:3]
    eef_ang = action_dict['cartesian_position'][3:6]
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    eef_pos_vel = action_dict['cartesian_velocity'][:3]
    eef_ang_vel = action_dict['cartesian_velocity'][3:6]
    joint_pos = action_dict['joint_position']
    joint_vel = action_dict['joint_velocity']
    grip_pos = action_dict['gripper_position']
    grip_vel = action_dict['gripper_velocity']

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    
    arm_action = tf.concat([eef_pos, eef_ang, eef_pos_vel, eef_ang_vel, joint_pos, joint_vel, grip_pos, grip_vel], axis=0)
    action['arm_concat'] = arm_action
    action['terminate'] = step['is_terminal']

    # Write the action format
    action['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw,arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel,gripper_joint_0_pos,gripper_joint_0_vel")

    # Convert raw state to our state
    # Robot state
    state = step['observation']
    eef_pos = state['cartesian_position'][:3]
    eef_ang = state['cartesian_position'][3:6]
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    joint_pos = state['joint_position']
    grip_pos = 1 - state['gripper_position']

    # Concatenate the state
    state['arm_concat'] = tf.concat([
        joint_pos,grip_pos,eef_pos,eef_ang], axis=0)
    

    # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_joint_0_pos,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
    pass
