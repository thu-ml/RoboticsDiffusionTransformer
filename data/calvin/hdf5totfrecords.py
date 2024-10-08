import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bool_feature(value):
    """Returns a bool_list from a boolean."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def serialize_example(action, robot_obs, rgb_static, rgb_gripper, instruction, terminate_episode):
    # Feature for fixed-length fields
    feature = {
        'action': _bytes_feature(tf.io.serialize_tensor(action)),
        'robot_obs': _bytes_feature(tf.io.serialize_tensor(robot_obs)),
        'rgb_static': _bytes_feature(tf.io.serialize_tensor(rgb_static)),
        'rgb_gripper': _bytes_feature(tf.io.serialize_tensor(rgb_gripper)),
        'terminate_episode': _bool_feature(terminate_episode),
        'instruction': _bytes_feature(instruction),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(root_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get the language annotation and corresponding indices
    f = np.load(os.path.join(root_dir, "lang_annotations/auto_lang_ann.npy"), allow_pickle=True)
    lang = f.item()['language']['ann']
    lang = np.array([x.encode('utf-8') for x in lang])
    lang_start_end_idx = f.item()['info']['indx']
    num_ep = len(lang_start_end_idx)
    
    with tqdm(total=num_ep) as pbar:
        for episode_idx, (start_idx, end_idx) in enumerate(lang_start_end_idx):
            pbar.update(1)

            step_files = [
                f"episode_{str(i).zfill(7)}.npz"
                for i in range(start_idx, end_idx + 1)
            ]
            action = []
            robot_obs = []
            rgb_static = []
            rgb_gripper = []
            instr = lang[episode_idx]
            for step_file in step_files:
                filepath = os.path.join(root_dir, step_file)
                f = np.load(filepath)
                # Get relevent things
                action.append(f['actions'])
                robot_obs.append(f['robot_obs'])
                rgb_static.append(f['rgb_static'])
                rgb_gripper.append(f['rgb_gripper'])
                
            tfrecord_path = os.path.join(out_dir, f'{episode_idx:07d}.tfrecord')
            print(f"Writing TFRecords to {tfrecord_path}")
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for i in range(len(step_files)):
                    serialized_example = serialize_example(
                        action[i], robot_obs[i], rgb_static[i], rgb_gripper[i], instr, i == len(step_files) - 1
                    )
                    writer.write(serialized_example)

output_dirs = [
    '../datasets/calvin/tfrecords/training',
    '../datasets/calvin/tfrecords/validation'
]

for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

root_dirs = [
    '../datasets/calvin/task_ABC_D/training',
    '../datasets/calvin/task_ABC_D/validation'
]

for root_dir, output_dir in zip(root_dirs, output_dirs):
    print(f"Writing TFRecords to {output_dir}")
    write_tfrecords(root_dir, output_dir)
