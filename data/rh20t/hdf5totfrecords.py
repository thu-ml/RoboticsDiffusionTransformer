import numpy as np
import os
import cv2
from multiprocessing import Pool, cpu_count, current_process
import tensorflow as tf
from tqdm import tqdm
import json

def _parse_function(proto):
    # Define how to parse the data here.
    feature_description = {
        'joint': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'instruction': tf.io.FixedLenFeature([], tf.string),
        'terminate_episode': tf.io.FixedLenFeature([], tf.int64),
        'gripper': tf.io.FixedLenFeature([], tf.string, default_value=""), 
        'tcp': tf.io.FixedLenFeature([], tf.string, default_value=""),
        'tcp_base': tf.io.FixedLenFeature([], tf.string, default_value="")
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    # Parse tensors
    parsed_features['joint'] = tf.io.parse_tensor(parsed_features['joint'], out_type=tf.float64)
    parsed_features['image'] = tf.io.parse_tensor(parsed_features['image'], out_type=tf.uint8)
    parsed_features['instruction'] = tf.io.parse_tensor(parsed_features['instruction'], out_type=tf.string)
    parsed_features['gripper'] = tf.cond(
        tf.math.equal(parsed_features['gripper'], ""),
        lambda: tf.constant([], dtype=tf.float64),
        lambda: tf.io.parse_tensor(parsed_features['gripper'], out_type=tf.float64)
    )
    parsed_features['tcp'] = tf.cond(
        tf.math.equal(parsed_features['tcp'], ""),
        lambda: tf.constant([], dtype=tf.float64),
        lambda: tf.io.parse_tensor(parsed_features['tcp'], out_type=tf.float64)
    )
    parsed_features['tcp_base'] = tf.cond(
        tf.math.equal(parsed_features['tcp_base'], ""),
        lambda: tf.constant([], dtype=tf.float64),
        lambda: tf.io.parse_tensor(parsed_features['tcp_base'], out_type=tf.float64)
    )
    return parsed_features

def convert_color(color_file, color_timestamps):
    """
    Args:
    - color_file: the color video file;
    - color_timestamps: the color timestamps;
    - dest_color_dir: the destination color directory.
    """
    cap = cv2.VideoCapture(color_file)
    cnt = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (640, 360))
            frames.append(resized_frame)
            cnt += 1
        else:
            break
    cap.release()
    return frames
    
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bool_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def serialize_example(joint,gripper,tcp,tcp_base,image,instruction,terminate_episode):
    feature = {
            'joint': _bytes_feature(tf.io.serialize_tensor(joint)),
            'image': _bytes_feature(tf.io.serialize_tensor(image)),
            'instruction': _bytes_feature(tf.io.serialize_tensor(instruction)),
            'terminate_episode': _bool_feature(terminate_episode),
        }
    if gripper is not None:
        feature['gripper'] = _bytes_feature(tf.io.serialize_tensor(gripper))
    if tcp is not None:
        feature['tcp'] = _bytes_feature(tf.io.serialize_tensor(tcp))
    if tcp_base is not None:
        feature['tcp_base'] = _bytes_feature(tf.io.serialize_tensor(tcp_base))
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def compress_tfrecord(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    # Serialize and write to a new TFRecord file
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for features in parsed_dataset:
            image_tensor = features['image']
            image_np = image_tensor.numpy()  
            if len(image_np.shape) <= 1: # already compressed
                return
            _, compressed_image = cv2.imencode('.jpg', image_np)
            features['image'] = tf.io.serialize_tensor(tf.convert_to_tensor(compressed_image.tobytes(), dtype=tf.string))

            def _bytes_feature(value):
                """Returns a bytes_list from a string / byte."""
                if isinstance(value, type(tf.constant(0))):
                    value = value.numpy()  
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            feature_dict = {
                'joint': _bytes_feature(features['joint']),
                'image': _bytes_feature(features['image']),
                'instruction': _bytes_feature(features['instruction']),
                'terminate_episode': tf.train.Feature(int64_list=tf.train.Int64List(value=[features['terminate_episode']])),
                'gripper': _bytes_feature(features['gripper']),
                'tcp': _bytes_feature(features['tcp']),
                'tcp_base': _bytes_feature(features['tcp_base'])
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            serialized_example = example_proto.SerializeToString()
            writer.write(serialized_example)
    print(f"compressed {tfrecord_path}")
    
def write_task(args):
    task_dir,output_dir = args
    
    all_instructions = json.load(open('./instruction.json'))
    instruction = None
    for taskid in list(all_instructions.keys()):
        if taskid in task_dir:
            instruction = all_instructions[taskid]['task_description_english']
    if instruction is None:
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joints = np.load(os.path.join(task_dir,"transformed/joint.npy"),allow_pickle=True).item()
    if not os.path.exists(os.path.join(task_dir,"transformed/gripper.npy")):
        return
    grippers = np.load(os.path.join(task_dir,"transformed/gripper.npy"),allow_pickle=True).item()
    tcps = np.load(os.path.join(task_dir,"transformed/tcp.npy"),allow_pickle=True).item()
    tcp_bases = np.load(os.path.join(task_dir,"transformed/tcp_base.npy"),allow_pickle=True).item()
    
    for camid in joints.keys():
        timesteps = joints[camid]
        if len(timesteps) == 0:
            continue
        tfrecord_path = os.path.join(output_dir,f'cam_{camid}.tfrecord')
        timesteps_file = os.path.join(task_dir,f'cam_{camid}/timestamps.npy')
        
        if not os.path.exists(timesteps_file):
            continue
        if os.path.exists(tfrecord_path) and os.path.getsize(tfrecord_path) > 0:
            continue

        timesteps_file = np.load(timesteps_file,allow_pickle=True).item()
        images = convert_color(os.path.join(task_dir,f'cam_{camid}/color.mp4'),timesteps_file['color'])
        if len(timesteps) != len(images): ## BUG FROM RH20T
            continue
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for i,timestep in enumerate(timesteps):
                # image = cv2.imread(os.path.join(img_dir,f"{timestep}.jpg"))
                image = cv2.imencode('.jpg', images[i])[1].tobytes()
                joint_pos = joints[camid][timestep]
                tcp = next((item for item in tcps[camid] if item['timestamp'] == timestep), None)['tcp']
                tcp_base = next((item for item in tcp_bases[camid] if item['timestamp'] == timestep), None)['tcp']
                if timestep not in grippers[camid]:
                    gripper_pos = None
                else:
                    gripper_pos = grippers[camid][timestep]['gripper_info']
                terminate_episode = i == len(timesteps) - 1
                # read from instruction.json
                serialized_example = serialize_example(joint_pos,gripper_pos,tcp,tcp_base,image,instruction,terminate_episode)
                writer.write(serialized_example)
            

def write_tfrecords(root_dir,output_dir,num_processes = None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if num_processes is None:
        num_processes = cpu_count()
        
    num_files = 0
    args = []
    for dirs in os.listdir(root_dir):
        for task in os.listdir(os.path.join(root_dir,dirs)):
            if 'human' in task:
                continue
            task_dir = os.path.join(root_dir,dirs,task)
            joint_path = os.path.join(task_dir,"transformed/joint.npy")
            if not os.path.exists(joint_path):
                continue
            num_files += 1
            task_out = os.path.join(output_dir,dirs,task)
            os.makedirs(task_out,exist_ok=True)
            args.append((task_dir,task_out))
            
    with tqdm(total=num_files, desc="Processing files") as pbar:
        with Pool(num_processes) as pool:   
            for _ in pool.imap_unordered(write_task, args):
                pbar.update(1)
                
write_tfrecords('../datasets/rh20t/raw_data/','../datasets/rh20t/tfrecords/')
