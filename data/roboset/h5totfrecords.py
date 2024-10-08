import tensorflow as tf
import h5py
import os
import fnmatch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bool_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def serialize_example(action, action_gripper, qpos, qvel, qpos_gripper, qvel_gripper, rgb_left, rgb_right, rgb_top, terminate_episode):
    feature = {
        'action': _bytes_feature(tf.io.serialize_tensor(action)),
        'action_gripper': _bytes_feature(tf.io.serialize_tensor(action_gripper)),
        'qpos': _bytes_feature(tf.io.serialize_tensor(qpos)),
        'qvel': _bytes_feature(tf.io.serialize_tensor(qvel)),
        'qpos_gripper': _bytes_feature(tf.io.serialize_tensor(qpos_gripper)),
        'qvel_gripper': _bytes_feature(tf.io.serialize_tensor(qvel_gripper)),
        'rgb_left': _bytes_feature(tf.io.serialize_tensor(rgb_left)),
        'rgb_right': _bytes_feature(tf.io.serialize_tensor(rgb_right)),
        'rgb_top': _bytes_feature(tf.io.serialize_tensor(rgb_top)),
        'terminate_episode': _bool_feature(terminate_episode),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def process_file(params):
    filepath, output_dir = params
    with h5py.File(filepath, 'r') as f:
        for Trial in f.keys():
            data = f[Trial]['data']
            tfrecord_path = os.path.join(output_dir, os.path.basename(filepath).replace('.h5', f'_{Trial}.tfrecord'))
            if os.path.exists(tfrecord_path) and os.path.getsize(tfrecord_path) > 0:
                continue
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                num_episodes = data['ctrl_arm'].shape[0]
                for i in range(num_episodes):
                    action = data['ctrl_arm'][i]
                    action_gripper = data['ctrl_ee'][i]
                    qpos = data['qp_arm'][i]
                    qvel = data['qv_arm'][i]
                    qpos_gripper = data['qp_ee'][i]
                    qvel_gripper = data['qv_ee'][i]
                    rgb_left = data['rgb_left'][i]
                    rgb_right = data['rgb_right'][i]
                    rgb_top = data['rgb_top'][i]
                    terminate_episode = i == num_episodes - 1
                    serialized_example = serialize_example(action, action_gripper, qpos, qvel, qpos_gripper, qvel_gripper, rgb_left, rgb_right, rgb_top, terminate_episode)
                    writer.write(serialized_example)

def write_tfrecords(root_dir, out_dir, num_processes=None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if num_processes is None:
        num_processes = cpu_count()

    file_list = []
    num_files = 0
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, '*.h5'):
            filepath = os.path.join(root, filename)
            output_dir = os.path.join(out_dir, os.path.relpath(os.path.dirname(filepath), root_dir))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            num_files += 1
            file_list.append((filepath, output_dir))

    with tqdm(total=num_files, desc="Processing files") as pbar:
        with Pool(num_processes) as pool:
            for _ in pool.imap_unordered(process_file, file_list):
                pbar.update(1)

root_dir = '../datasets/roboset/'
output_dir = '../datasets/roboset/tfrecords/'

write_tfrecords(root_dir, output_dir)
