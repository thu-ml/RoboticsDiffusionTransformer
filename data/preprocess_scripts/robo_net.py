import tensorflow as tf
import numpy as np

from data.utils import clean_task_instruction, euler_to_quaternion, euler_to_rotation_matrix, \
    rotation_matrix_to_ortho6d


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    action = step['action']
    eef_delta_pos = action[:3]
    eef_delta_angle_yaw = action[3:4]
    eef_ang = tf.stack([0.0, 0.0, eef_delta_angle_yaw[0]], axis=0)
    eef_ang = euler_to_quaternion(eef_ang)
    eef_gripper_open = (1 - action[4:5]) / 2
    
    step['action'] = {}
    action = step['action']
    action['terminate'] = step['is_terminal']

    # No base found

    # Concatenate the action
    arm_action = tf.concat([eef_delta_pos, eef_ang, eef_gripper_open], axis=0)
    action['arm_concat'] = arm_action

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")

    # Convert raw state to our state
    state = step['observation']
    eef_pos = state['state'][:3]
    eef_ang_yaw = state['state'][3:4]
    eef_ang = tf.stack([0.0, 0.0, eef_ang_yaw[0]], axis=0)
    eef_ang = euler_to_rotation_matrix(eef_ang)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    grip_joint_pos = state['state'][4:5]
    # If abs(grip_joint_pos) > 3.15, then convert it to the radian
    grip_joint_pos = tf.cond(tf.greater(tf.abs(grip_joint_pos), 3.15), 
                             lambda: grip_joint_pos / 180 * np.pi, 
                             lambda: grip_joint_pos)
    # Concatenate the state
    state['arm_concat'] = tf.concat([eef_pos,eef_ang,grip_joint_pos],axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,gripper_open")

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
