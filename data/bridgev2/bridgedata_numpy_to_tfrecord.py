"""
Converts data from the BridgeData numpy format to TFRecord format.

Consider the following directory structure for the input data:

    bridgedata_numpy/
        rss/
            toykitchen2/
                set_table/
                    00/
                        train/
                            out.npy
                        val/
                            out.npy
        icra/
            ...

The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=5, then --input_path should be
"bridgedata_numpy", and all data will be processed. If --depth=3, then
--input_path should be "bridgedata_numpy/rss/toykitchen2", and only data
under "toykitchen2" will be processed.

The same directory structure will be replicated under --output_path.  For
example, in the second case, the output will be written to
"{output_path}/set_table/00/...".

Can read/write directly from/to Google Cloud Storage.

Written by Kevin Black (kvablack@berkeley.edu).
"""
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
import pickle
from multiprocessing import cpu_count

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse. Looks for {input_path}/dir_1/dir_2/.../dir_{depth-1}/train/out.npy",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
num_workers = 8
flags.DEFINE_integer("num_workers", num_workers, "Number of threads to use")

print(f"using {num_workers} workers")

def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )
    
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _strings_feature(string_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=s.encode('utf-8')))

def _bool_feature(value):
    """Returns a bool_list from a boolean."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def process(path):
    # with tf.io.gfile.GFile(path, "rb") as f:
    #     arr = np.load(f, allow_pickle=True)
    try:
        with tf.io.gfile.GFile(path, "rb") as f:
            arr = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return
    
    dirname = os.path.dirname(os.path.abspath(path))
    outpath = os.path.join(FLAGS.output_path, *dirname.split(os.sep)[-FLAGS.depth :])

    if tf.io.gfile.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    if len(arr) == 0:
        logging.info(f"Skipping {path}, empty")
        return
    
    tf.io.gfile.makedirs(outpath)

    for i,traj in enumerate(arr):
        write_path = f"{outpath}/out_{i}.tfrecord"
        with tf.io.TFRecordWriter(write_path) as writer:
            truncates = np.zeros(len(traj["actions"]), dtype=np.bool_)
            truncates[-1] = True
            frames_num = len(traj["observations"])
            # remove empty string 
            traj["language"] = [x for x in traj["language"] if x != ""]
            if len(traj["language"]) == 0:
                traj["language"] = [""]
            instr = traj["language"][0]
            if(len(traj["language"]) > 2):
                print(len(traj["language"]))
            for i in range(frames_num):
                tf_features = {
                    "observations/images0": tensor_feature(
                        np.array(
                            [traj["observations"][i]["images0"]],
                            dtype=np.uint8,
                        )
                    ),
                    "observations/state": tensor_feature(
                        np.array(
                            [traj["observations"][i]["state"]],
                            dtype=np.float32,
                        )
                    ),
                    "observations/qpos": tensor_feature(
                        np.array(
                            [traj["observations"][i]["qpos"]],
                            dtype=np.float32,
                        )
                    ),
                    "observations/eef_transform": tensor_feature(
                        np.array(
                            [traj["observations"][i]["eef_transform"]],
                            dtype=np.float32,
                        )
                    ),
                    "language": _bytes_feature(instr),
                    "actions": tensor_feature(
                        np.array(traj["actions"][i], dtype=np.float32)
                    ),
                    "truncates": _bool_feature(i == frames_num - 1),
                }
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature = tf_features
                    )
                )
                writer.write(example.SerializeToString())


def main(_):
    assert FLAGS.depth >= 1

    paths = tf.io.gfile.glob(
        tf.io.gfile.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1)))
    )
    paths = [f"{p}/train/out.npy" for p in paths] + [f"{p}/val/out.npy" for p in paths]
    # num_episodes = 0
    # for dirpath in paths:
    #     with tf.io.gfile.GFile(dirpath, "rb") as f:
    #         arr = np.load(dirpath, allow_pickle=True)
    #     num_episodes += len(arr)
    # print(num_episodes)
    with Pool(FLAGS.num_workers) as p:
        list(tqdm.tqdm(p.imap(process, paths), total=len(paths)))


if __name__ == "__main__":
    app.run(main)
