import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_rotation_matrix, rotation_matrix_to_ortho6d


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    action = step['action']
    eef_pos = action[:3]
    eef_quat = action[3:7]
    eef_ang = quaternion_to_rotation_matrix(eef_quat)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    grip_pos = action[7:]

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    action['arm_concat'] = tf.concat([
        grip_pos,eef_pos,eef_ang], axis=0)
    # Write the action format
    action['format'] = tf.constant(
        "gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")
    action['terminate'] = step['is_terminal']

    # Convert raw state to our state
    state = step['observation']
    robot_state = state['state']
    eef_pos = robot_state[:3]
    eef_quat = robot_state[3:7]
    eef_ang = quaternion_to_rotation_matrix(eef_quat)
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    joint_pos = robot_state[7:14]
    grip_pos = robot_state[14:16] * 24.707  # rescale to [0, 1]
    joint_vel = robot_state[16:23]

    # Concatenate the state
    state['arm_concat'] = tf.concat([
        grip_pos,eef_pos,eef_ang,joint_pos,joint_vel], axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "gripper_joint_0_pos,gripper_joint_1_pos,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5,arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,arm_joint_0_vel,arm_joint_1_vel,arm_joint_2_vel,arm_joint_3_vel,arm_joint_4_vel,arm_joint_5_vel,arm_joint_6_vel")

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
