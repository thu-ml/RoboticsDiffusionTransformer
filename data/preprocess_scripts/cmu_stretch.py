import tensorflow as tf

from data.utils import clean_task_instruction, quaternion_to_euler,euler_to_quaternion
def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(tf.equal(terminate_act, tf.constant(0.0, dtype=tf.float32)),tf.constant(False),tf.constant(True))

def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action

    origin_action = step['action']
    step['action']={}
    action=step['action']
    action['terminate']=terminate_act_to_bool(origin_action[7])
    
    
    eef_pos=origin_action[:3]
    eef_ang=origin_action[3:6]
    eef_ang = euler_to_quaternion(eef_ang)
    grip_open=origin_action[6:7]
    # No base found

    # Concatenate the action
    action['arm_concat'] = tf.concat([eef_pos,eef_ang,grip_open],axis=0)

    # Write the action format
    action['format'] = tf.constant(
        "eef_delta_pos_x,eef_delta_pos_y,eef_delta_pos_z,eef_delta_angle_x,eef_delta_angle_y,eef_delta_angle_z,eef_delta_angle_w,gripper_open")
    
    # Convert raw state to our state
    state = step['observation']
    # Concatenate the state
    eef_pos_x = state['state'][0:1]
    eef_pos_z = state['state'][2:3]
    grip_open = state['state'][3:4]
    state['arm_concat'] = tf.concat(
        [eef_pos_x, eef_pos_z, grip_open], axis=0)
    # Write the state format
    state['format'] = tf.constant(
        "eef_pos_x,eef_pos_z,gripper_open")

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
    DATASET_NAME = 'cmu_stretch'
    # Load the dataset
    dataset = tfds.builder_from_directory(
        builder_dir=dataset_to_path(
            DATASET_NAME, DATASET_DIR))
    dataset = dataset.as_dataset(split='all')

    # Inspect the dataset
    for episode in dataset:
        for step in episode['steps']:
            print(step['action'][6:7])

