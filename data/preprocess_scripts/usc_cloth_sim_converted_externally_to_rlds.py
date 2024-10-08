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
    # Robot action, consists of x,y,z goal and picker commandpicker<0.5 = open, picker>0.5 = close.
    action = step['action']
    eef_delta_pos = action[:3]
    grip_open = tf.cast(tf.expand_dims(action[3] < 0.5, axis=0), dtype=tf.float32)

    # Concatenate the action
    step['action'] = {}
    action = step['action']
    
    action['terminate'] = step['is_terminal']
    action['arm_concat'] = tf.concat([eef_delta_pos, grip_open], axis=0)
   
    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,gripper_open")

    # State doesnt exist in this dataset

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
