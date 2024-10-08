import tensorflow as tf

from data.utils import clean_task_instruction

def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action

    origin_action = step['action']
    step['action']={}
    action=step['action']
    
    eef_delta_pos=origin_action
    # No base found

    # Concatenate the action
    action['arm_concat'] = eef_delta_pos
    action['terminate'] = step['is_terminal']

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y")
    
    # Convert raw state to our state
    state = step['observation']
    # Concatenate the state
    eef_pos=state['effector_translation']
    state['arm_concat'] = eef_pos
    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_y")

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
    instr = step['observation']['instruction']
    # Convert bytes to tf.string
    instr = tf.strings.unicode_encode(instr, 'UTF-8')
    # Remove '\x00'
    instr = tf.strings.regex_replace(instr, '\x00', '')
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr
    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    DATASET_DIR = 'data/datasets/openx_embod'
    DATASET_NAME = 'language_table'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step)

