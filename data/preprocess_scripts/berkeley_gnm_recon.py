import tensorflow as tf

from data.utils import clean_task_instruction, euler_to_quaternion, euler_to_rotation_matrix,\
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
    action['terminate'] = step['is_terminal']
  
    eef_pos=tf.cast(origin_action,dtype=tf.float32)
    eef_ang=tf.cast(step['action_angle'][2:3],dtype=tf.float32)
    eef_ang = euler_to_quaternion(tf.stack([0,0,eef_ang[0]],axis=0))
    # No base found

    # Concatenate the action
    action['arm_concat'] = tf.concat([eef_pos,eef_ang],axis=0)

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w")
    
    # Convert raw state to our state
    state = step['observation']
    # Concatenate the state
    eef_pos=tf.cast(state['position'],dtype=tf.float32)
    eef_ang=tf.cast(state['yaw'],dtype=tf.float32)
    eef_ang = euler_to_rotation_matrix(tf.stack([0,0,eef_ang[0]],axis=0))
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    state['arm_concat'] = tf.concat([eef_pos/100,eef_ang],axis=0)
    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_angle_x,eef_angle_y,eef_angle_z,eef_angle_w")

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
    DATASET_NAME = 'berkeley_gnm_recon'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step['action'][6:7])

