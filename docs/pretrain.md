# Pipeline of Pre-Training RDT

Firstly, you need to install the prerequisites for RDT (see [README](../README.md#installation)). Then, you can install the prerequisites for TensorFlow Dataset (in another Conda environment).

## Installation for TensorFlow Dataset

```bash
# Under the root directory of this repo
conda create -n rdt-data python=3.10
conda activate rdt-data

# Install all the prequisites
pip install -r requirements_data.txt
# Or you can manually install each package (please refer to requirements_data.txt for specific versions) 
pip install tfds-nightly gsutil tensorflow Pillow pyyaml opencv-python tensorflow-graphics imageio[ffmpeg]
# If the speed is too slow, you can specify alternative sources (tfds-nightly is not available in Tsinghua mirror)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gsutil tensorflow Pillow pyyaml opencv-python tensorflow-graphics imageio[ffmpeg]
```

## Download and Prepare Datasets

We introduce how to download each of our pre-training datasets. If you plan to pre-train on a subset of them, just download the ones you need. You can also fine-tune RDT through this pipeline only if your target dataset is included below or in the Google Cloud Storage.

|  Dataset    |   Sample Percentage (%)   |
| ---- | ---- |
| RT-1 Dataset | 9.00 |
| TACO Dataset | 1.99 |
| JACO Play Dataset | 1.10 |
| Cable Routing Dataset | 0.27 |
| NYU Door Opening | 0.33 |
| Viola | 0.40 |
| Berkeley UR5 | 1.06 |
| TOTO | 1.06 |
| Kuka | 1.66 |
| Language Table | 3.32 |
| Columbia Cairlab Pusht Real | 0.40 |
| Stanford Kuka Multimodal Dataset | 1.83 |
| Stanford Hydra Dataset  | 0.80 |
| Austin Buds Dataset | 0.23 |
| Maniskill Dataset | 5.78 |
| Furniture Bench Dataset | 2.36 |
| UCSD Kitchen Dataset | 0.40 |
| UCSD Pick And Place Dataset | 1.23 |
| Austin Sailor Dataset | 0.50 |
| Austin Sirius Dataset | 0.80 |
| BC Z | 6.91 |
| UTokyo PR2 Opening Fridge | 0.30 |
| UTokyo PR2 Tabletop Manipulation | 0.50 |
| UTokyo Xarm Pick And Place | 0.33 |
| UTokyo Xarm Bimanual | 0.03 |
| Berkeley MVP | 0.73 |
| Berkeley RPT | 1.00 |
| KAIST Nonprehensile | 0.46 |
| Tokyo U LSMO | 0.23 |
| DLR Sara Grid Clamp | 0.03 |
| Robocook | 1.66 |
| Imperialcollege Sawyer Wrist Cam | 0.43 |
| Iamlab CMU Pickup Insert | 0.83 |
| UTAustin Mutex | 1.29 |
| Fanuc Manipulation | 0.66 |
| Play Fusion | 0.80 |
| Droid | 10.06 |
| FMB| 1.39 |
| Dobb·E | 1.20 |
| QUT Dexterous Manipulation | 0.46 |
| Aloha Dataset | 4.98 |
| Mobile Aloha Dataset | 4.98 |
| Roboset | 4.48 |
| RH20T | 10.99 |
| Calvin Dataset | 3.32 |
| Bridgev2 | 7.44 |

Before everything, let's link the dataset directory on your disk to a subfolder of this repo:

```bash
ln -s /path/to/dataset /path/to/repo/RoboticsDiffusionTransformer/data/datasets
```

### Open X-Embodiment

Specify the correct path to the `gsutil` in your Conda in [this file](../data/openx_embod/download.sh#L72).

Run the following commands to download our selected datasets for the Open X-Embodiment:

```bash
# Under the root directory of this repo
cd data/openx_embod
# Download all datasets
bash download_openx_embod.sh
```

Note: By modifying `download_openx_embod.sh`,  you can download any dataset on the Google Cloud (as long as it can be downloaded with `gsutil` and is stored in `TFRecord` format), not just the ones we have listed.

### Mobile ALOHA Dataset

Download the Mobile ALOHA Dataset from the [official website](https://mobile-aloha.github.io) to `data/datasets/aloha`, then run:

```bash
cd data/aloha
# Convert the dataset to TFRecord
python hdf5totfrecords.py
```

### Bridgev2

Run:

```bash
cd data/bridgev2
# Download and preprocess the dataset
sh download.sh
```

### Calvin

Run:

```bash
cd data/calvin
# Download and preprocess the dataset
sh download.sh
# Convert the dataset to TFRecord format
python hdf5totfrecords.py
```

### RH20T

Download the RH20T Dataset from there [official website](https://rh20t.github.io/#download) to `data/datasets/rh20t`, then run

```bash
cd data/rh20t
# Convert the dataset to TFRecord
python hdf5totfrecords.py
```

### RoboSet

Run:

```bash
cd data/roboset
# Download and preprocess the dataset
sh download.sh
```

## If Want to Train on a New Dataset


If you want to train on a new dataset (e.g., `my_pretrain_dataset`) through this pre-training pipeline, you need to modify several files as follows:

##### 1. `configs/dataset_control_freq.json`

Add the control frequency of your dataset.

##### 2. `data/preprocess_scripts/my_pretrain_dataset.py`

If your dataset can be loaded by `tfds.builder_from_directory()`, then you only need to download it into the folder of Open X-Embodiment `data/datasets/openx_embod` and implement the function of `process_step()`. You may need to specify the tfds loading path in L78 (see [this file](../data/vla_dataset.py#L78)). We refer to `data/preprocess_scripts/droid.py` for an example.

If not, you need to first convert it into TFRecords and then implement both `load_dataset()` and `process_step()`. We refer to `data/agilex/hdf5totfrecords.py` and `data/preprocess_scripts/agilex.py` for examples.

Here some descriptions:

##### `load_dataset(seed: int)`

- Returns a dataset that supports iterator and `repeat` method with a random seed.
- Suggested implementation: Use `tf.data.Dataset.from_generator` and `tf.data.TFRecordDataset`.
- The iterator should return a subdataset that supports iterator representing one episode with the following structure:
  - `step`: A dataset object that supports iterator containing multiple frames per episode.
    - `observation`: A dictionary containing your images.
      - `your_first_image_key`: Your observation RGB image keys.
      - ...
    - `other_attribute`: Any other relevant attributes.

##### `process_step(step: dict) -> dict`

Processes a single frame and returns a dictionary with the following keys:

- `observation`:
  - `your_first_view_image: tf.Tensor`: Your first view image.
  - `arm_concat: tf.Tensor`: Concatenation of physical states.
  - `format: tf.constant(string)`: Format of `arm_concat` (e.g., `arm_joint_pos_0,arm_joint_pos_1,arm_joint_pos_2`).
- `action`: Frame action (leave empty if there's none).
  - `arm_concat`: Same as in `observation`.
  - `format`: Same as in `observation`.
  - `terminate: tf.Tensor`: Boolean Tensor indicates if the episode ends.

**IMPORTANT**: You should only use TensorFlow functions for any branch or loop operations. For example, use `tf.cond` instead of `if`.

##### 3. `configs/dataset_img_keys.json`

Add the image keys of your dataset. For example:

```json
"my_pretrain_dataset": {
  "image_keys": [
    "exterior-cam",
    "right-wrist-cam",
    "left-wrist-cam",
    "left-wrist-cam"
  ],
  "image_mask": [1, 1, 1, 0]
}
```

- To make TensorFlow happy, you have to specify four images in this order: `exterior-cam, right-wrist-cam, left-wrist-cam, any-cam`. Each key should correspond to your `step` attribute key of observation images.

- If you only have a single wrist, just make it a *right* wrist.

- The `image_mask` indicates whether each image is valid (1) or not (0).

- What if you don’t have four images? Simply repeat the images in the following positions and set their masks to 0 (invalid).

- The key order is *strict*. If you don't have the exterior camera but have both wrists, leave the exterior position blank (or pad) and use the following:

   ```json
   "my_pretrain_dataset": {
     "image_keys": [
       "right-wrist-cam",
       "right-wrist-cam",
       "left-wrist-cam",
       "left-wrist-cam"
     ],
     "image_mask": [0, 1, 1, 0]
   }
   ```

- During training, only the first *three* cameras will be used.
##### 4. `configs/dataset_stat.json`

Compute the statistics (min, max, mean, and std) for your dataset:

```bash
# Use -h to see the full usage
python -m data.compute_dataset_stat --skip_exist
```
This will update the `dataset_stat.json` file with your dataset's statistics.

##### 5. `data/vla_dataset.py`

- Add your dataset to `DATASET_NAMES_NOOPENX` if it cannot be loaded by `tfds.builder_from_directory()`.
- If your dataset only contains action but no proprioception (i.e., robot state), add your dataset to `DATASET_NAMES_NO_STATE` in [this file](../data/preprocess.py).
- Normally, we consider the future state as the action of current timestep. If you want to use different actions, you should implement more functions. We refer to `flatten_episode_agilex()` in [this file](../data/episode_transform.py) and `_generate_json_state_agilex()` in [this file](../data/preprocess.py) for examples. You may also refer to L318 in [this file](../data/preprocess.py) and L128 in [this file](../data/vla_dataset.py) for how to select your dataset and preprocess it differently.

## Start Pre-Training

We employ a producer-consumer framework with TensorFlow Dataset for fast data loading. Since most of the datasets in the Open X-Embodiment are stored in the form of `TFRecord`, we convert all pre-training datasets into `TFRecord` for storage. In pre-training, we use the producer process to decompress the data from `TFRecord` and store it in a buffer on the hard disk. At the same time, we use the consumer process to read data from the buffer in a disorderly order and feed it to the model training.  This not only decouples the `TensorFlow` and `PyTorch` environments but also alleviates the training performance loss caused by the small size of the shuffling buffer in the memory.

[This file](../configs/base.yaml) includes configurations relevant to model architecture (including number of heads, hidden dimension, and so on) and data processing. You may need to modify `buf_path` (L22) to your real buffer path. This buffer is used as disk shuffling buffer for data loading. 

Configurations relevant to training are passed through *Command Line Arguments*. Use `python main.py -h ` to see the descriptions. We provide an example pre-training script in [this file](../pretrain.sh) (`pretrain.sh`). You may need to modify some of the parameters in this file, such as `CUTLASS_PATH` and `WANDB_PROJECT`.

You may need to modify the list of pre-training datasets in [this file](../configs/pretrain_datasets.json) and their corresponding sampling weights in [this file](../configs/pretrain_sample_weights.json). If you want to fine-tune RDT through this pipeline, you may need to remove abundant datasets in the list.

Before start pre-training, we first start the data producer process (if you use multiple nodes, you should run this command in each node):

```bash
# Under the root directory of this repo
conda activate rdt-data
# Use -h to see the full usage
python -m data.producer --fill_up
# Please proceed to the next step AFTER finishing the filling up process
```

Then, we run the pre-training script:

```bash
source pretrain.sh
```

Note: You can monitor the training process by observing `loss` (through a long window moving average), `overall_avg_sample_mse`, and the sampling MSE of each dataset in [Wandb](https://wandb.ai/site) or [TensorBoard](https://www.tensorflow.org/tensorboard). We empirically found that the lower the `overall_avg_sample_mse`, the better the model performs.
