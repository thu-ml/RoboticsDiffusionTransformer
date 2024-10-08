import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_rotation_matrix, rotation_matrix_to_ortho6d


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
    
    eef_delta_pos = action['world_vector']
    eef_ang=action['rotation_delta']

    # No gripper_open found
    # No base found
    
    # Concatenate the action
    arm_action=tf.concat([eef_delta_pos,eef_ang],axis=0)
    action['arm_concat']=arm_action
    #base_action = tf.concat([base_pos, base_ang], axis=0)
    #action['base_concat'] = base_action

    # Write the action format
    action['format']=tf.constant("eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw")
    
    # Convert raw state to our state
    state = step['observation']
    eef_pos = state['robot_state'][:3]
    eef_ang = quaternion_to_rotation_matrix(state['robot_state'][3:])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    
    # Concatenate the state
    state['arm_concat']=tf.concat([eef_pos,eef_ang],axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

    # Define the task instruction
    step['observation']['natural_language_instruction'] = tf.constant(
        "Route cable through the tight-fitting clip mounted on the table.")

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod/'
    DATASET_NAME = 'berkeley_cable_routing'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
