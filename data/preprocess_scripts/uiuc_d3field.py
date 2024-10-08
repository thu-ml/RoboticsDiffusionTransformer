import tensorflow as tf

from data.utils import clean_task_instruction, rotation_matrix_to_ortho6d

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
    # Robot displacement from last frame
    eef_delta_pos = origin_action[:3]

    # No base found

    # Concatenate the action
    action['arm_concat'] = eef_delta_pos

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z")

    # Convert raw state to our state
    state = step['observation']
    # Concatenate the state
    # 4x4 Robot end-effector state
    eef_mat=state['state']
    eef_pos=eef_mat[:3,3]
    rotation_matrix=eef_mat[:3,:3]
    eef_ang = rotation_matrix_to_ortho6d(rotation_matrix)
    state['arm_concat'] = tf.concat([eef_pos,eef_ang],axis=0)

    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

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
    DATASET_NAME = 'uiuc_d3field'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)
