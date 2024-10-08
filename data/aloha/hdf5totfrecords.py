import tensorflow as tf
import h5py
import os
import fnmatch
import cv2
import numpy as np
from tqdm import tqdm

def decode_img(img):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def decode_all_imgs(imgs):
    return [decode_img(img) for img in imgs]

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _bool_feature(value):
    """Returns a bool_list from a boolean."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def serialize_example(action, base_action, qpos, qvel, cam_high, cam_left_wrist, cam_right_wrist, cam_low, instruction, terminate_episode):
    if base_action is not None:
        feature = {
            'action': _bytes_feature(tf.io.serialize_tensor(action)),
            'base_action': _bytes_feature(tf.io.serialize_tensor(base_action)),
            'qpos': _bytes_feature(tf.io.serialize_tensor(qpos)),
            'qvel': _bytes_feature(tf.io.serialize_tensor(qvel)),
            'cam_high': _bytes_feature(tf.io.serialize_tensor(cam_high)),
            'cam_left_wrist': _bytes_feature(tf.io.serialize_tensor(cam_left_wrist)),
            'cam_right_wrist': _bytes_feature(tf.io.serialize_tensor(cam_right_wrist)),
            'instruction': _bytes_feature(instruction),
            'terminate_episode': _bool_feature(terminate_episode)
        }
    else:
        feature = {
            'action': _bytes_feature(tf.io.serialize_tensor(action)),
            'qpos': _bytes_feature(tf.io.serialize_tensor(qpos)),
            'qvel': _bytes_feature(tf.io.serialize_tensor(qvel)),
            'cam_high': _bytes_feature(tf.io.serialize_tensor(cam_high)),
            'cam_left_wrist': _bytes_feature(tf.io.serialize_tensor(cam_left_wrist)),
            'cam_right_wrist': _bytes_feature(tf.io.serialize_tensor(cam_right_wrist)),
            'cam_low': _bytes_feature(tf.io.serialize_tensor(cam_low)),
            'instruction': _bytes_feature(instruction),
            'terminate_episode': _bool_feature(terminate_episode)
        }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecords(root_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    num_files = 0
    for root, dirs, files in os.walk(root_dir):
        num_files += len(fnmatch.filter(files, '*.hdf5'))
    with tqdm(total=num_files) as pbar:
        for root, dirs, files in os.walk(root_dir):
            for filename in fnmatch.filter(files, '*.hdf5'):
                filepath = os.path.join(root, filename)
                with h5py.File(filepath, 'r') as f:
                    if not 'instruction' in f:
                        continue
                    pbar.update(1)
                    output_dir = os.path.join(out_dir, os.path.relpath(root, root_dir))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print(f"Writing TFRecords to {output_dir}")
                    tfrecord_path = os.path.join(output_dir, filename.replace('.hdf5', '.tfrecord'))
                    with tf.io.TFRecordWriter(tfrecord_path) as writer:
                        num_episodes = f['action'].shape[0]
                        for i in range(num_episodes):
                            action = f['action'][i]
                            if 'base_action' in f:
                                base_action = f['base_action'][i]
                            else:
                                base_action = None
                            qpos = f['observations']['qpos'][i]
                            qvel = f['observations']['qvel'][i]
                            cam_high = decode_img(f['observations']['images']['cam_high'][i])
                            cam_left_wrist = decode_img(f['observations']['images']['cam_left_wrist'][i])
                            cam_right_wrist = decode_img(f['observations']['images']['cam_right_wrist'][i])
                            if 'cam_low' in f['observations']['images']:
                                cam_low = decode_img(f['observations']['images']['cam_low'][i])
                            else:
                                cam_low = None
                            instruction = f['instruction'][()]
                            terminate_episode = i == num_episodes - 1
                            serialized_example = serialize_example(action, base_action, qpos, qvel, cam_high, cam_left_wrist, cam_right_wrist, cam_low, instruction, terminate_episode)
                            writer.write(serialized_example)
                        print(f"TFRecords written to {tfrecord_path}")
    print(f"TFRecords written to {out_dir}")

root_dir = '../datasets/aloha/'
output_dir = '../datasets/aloha/tfrecords/'

write_tfrecords(root_dir, output_dir)
