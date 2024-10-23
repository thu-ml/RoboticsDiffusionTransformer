# RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation

### 📝[Paper](https://arxiv.org/pdf/2410.07864) | 🌍[Project Page](https://rdt-robotics.github.io/rdt-robotics/) | 🤗[Model](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | 🛢️[Data](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data)

![](./assets/head.png)

RDT-1B is a **1B**-parameter (*largest* to date) imitation learning **Diffusion Transformer** pre-trained on **1M+** (*largest* to date) multi-robot episodes. Given language instruction and RGB images of up to three views, RDT can predict the next $64$ robot actions. RDT is inherently compatible with **almost all kinds of modern mobile manipulators**, from single-arm to dual-arm, joint to EEF, position to velocity, and even with wheeled locomotion.

We have fine-tuned RDT on **6K+** (one of the *largest*) self-collected bimanual episodes and deployed it on the ALOHA **dual-arm** robot. It has achieved state-of-the-art performance in terms of dexterity, zero-shot generalizability, and few-shot learning. You can find Demo videos on our [project page](https://rdt-robotics.github.io/rdt-robotics/).

This repo is an official PyTorch implementation of RDT, containing:

- 🛠️Model [implementation](models/rdt_runner.py) of RDT
- 🤗1M-step [checkpoint](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) of RDT-1B pre-trained on multi-robot data
- 🤗500K-step [checkpoint](https://huggingface.co/robotics-diffusion-transformer/rdt-170m) of RDT-170M (RDT(small) in [ablation](https://arxiv.org/pdf/2410.07864))
- 📈Training and sampling [scripts](train/train.py) (with DeepSpeed)
- 🤖An [example](scripts/agilex_inference.py) of real-robot deployment

The following guides include the [installation](#installation), [fine-tuning](#fine-tuning-on-your-own-dataset), and [deployment](#deployment-on-real-robots). Please refer to [pre-training](docs/pretrain.md) for a detailed list of pre-training datasets and a pre-training guide.

## 📰 News
- [2024/10/23] 🔥 **RDT-170M** (Smaller) model is released, a more VRAM-friendly solution 🚀💻.

## Installation

1. Clone this repo and install prerequisites:

    ```bash
    # Clone this repo
    git clone git@github.com:thu-ml/RoboticsDiffusionTransformer.git
    cd RoboticsDiffusionTransformer
    
    # Create a Conda environment
    conda create -n rdt python=3.10.0
    conda activate rdt
    
    # Install pytorch
    # Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
    pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
    
    # Install packaging
    pip install packaging==24.0
    
    # Install flash-attn
    pip install flash-attn --no-build-isolation
    
    # Install other prequisites
    pip install -r requirements.txt
    ```

2. Download off-the-shelf multi-modal encoders:

   You can download the encoders from the following links:

   - `t5-v1_1-xxl`: [link](https://huggingface.co/google/t5-v1_1-xxl/tree/main)🤗
   - `siglip`: [link](https://huggingface.co/google/siglip-so400m-patch14-384)🤗

   And link the encoders to the repo directory:

   ```bash
   # Under the root directory of this repo
   mkdir -p google
   
   # Link the downloaded encoders to this repo
   ln -s /path/to/t5-v1_1-xxl google/t5-v1_1-xxl
   ln -s /path/to/siglip-so400m-patch14-384 google/siglip-so400m-patch14-384
   ```
3. Fill the missing argument in [this file](configs/base.yaml#L22):
   
   Note that this buffer will only be used during pre-training. See [this doc](docs/pretrain.md) for more details.
   ```
   # ...
   
   dataset:
   # ...
   # ADD YOUR buf_path: the path to the buffer (at least 400GB)
      buf_path: /path/to/buffer
   # ...
   ```

## Fine-Tuning on Your Own Dataset

If your fine-tuning dataset is in the [Open X-Embodiment](https://robotics-transformer-x.github.io/) or the collection of our pre-training datasets (see [this doc](docs/pretrain.md#download-and-prepare-datasets)), you can also fine-tune RDT through the pre-trained pipeline. You need to remove other redundant datasets in the parameters. We refer to [this guide](docs/pretrain.md) (pre-training).

1. Prepare your dataset:

   You need to download your dataset to the disk and give it a name `my_cool_dataset`.

   Then, you can link your dataset to the repo directory:

   ```bash
   # Under the root directory of this repo
   cd data
   mkdir -p datasets
   
   # Link the downloaded dataset to this repo
   ln -s /path/to/my_cool_dataset datasets/my_cool_dataset
   ```

2. Implement the dataset loader:

   You need to:

   1. Register the configuration of `my_cool_dataset`:

      Append the control frequency of `my_cool_dataset` in [this file](configs/dataset_control_freq.json). Write the name of `my_cool_dataset` in [this file](configs/finetune_datasets.json) and [this file](configs/finetune_sample_weights.json), where the value of the sampling weight doesn't matter since you only have one dataset. In these two files, we leave a placeholder of `agilex`; you can simply replace it with `my_cool_dataset`.

   2. Re-Implement the class of `HDF5VLADataset`:

      You can find this class in [this file](data/hdf5_vla_dataset.py). In this file, we provide an example of loading the fine-tuning dataset used in our paper (see [this link](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data)).

      To adapt it to your dataset, you need to: (a) modify the `HDF5_DIR` (directory to `my_cool_dataset`) and `DATASET_NAME` (should be `"my_cool_dataset"`) in L21 and L22; (b) Implement the two functions of `parse_hdf5_file()` and `parse_hdf5_file_state_only()`. Please take a look at the original file for detailed comments and examples.

      Note 1: Despite its name, you don't necessarily need to use HDF5 to store your data. Just make sure that the class is correctly implemented.

      Note 2: During implementation, you may need to fill your robot action into the unified action vector (L180-194). Please refer to [this file](configs/state_vec.py) for an explanation of each element in the unified vector. We have reserved enough slots for each physical quantity. For example, we have reserved ten slots for joint angles. If your robot arm has six degrees of freedom, you only need to fill in the first six. 

      **IMPORTANT 1:** If your robot is single-arm, please fill its action into the *right-arm* portion of the unified action vector, aligning with our pre-training datasets.

      **IMPORTANT 2:** We use [6D representation](https://arxiv.org/pdf/1812.07035) for EEF rotation. If your action space contains EEF rotation (angle or quaternion), please refer to [this file](docs/test_6drot.py) for conversion. We note that this mapping is not reversible. Different Euler angles may be equivalent and correspond to the same 6D representation.

      **IMPORTANT 3:** No physical quantities (except the gripper width) are normalized during pre-training. This can preserve each physical quantity's meaning, thereby promoting generalization across robots. Therefore, we encourage you not to normalize any physical quantities but to choose appropriate units for them. Generally, we use the International System of Units, which ensures that most values fall within [-1,1]. As an exception, we perform min-max normalization on the gripper width to [0,1].

      **IMPORTANT 4:** If you use RTX 4090 (or lower), the GPU memory may be too low to load the `t5-v1_1-xxl` encoder. Instead, we recommend you precompute the language embeddings (see [this file](scripts/encode_lang_batch.py) for an example script) and load them during training. In this way, you need to specify the path to the embeddings in the `HDF5VLADataset` (see L148) rather than the natural language.

   3. Compute the dataset statistics information for `my_cool_dataset`:

      ```bash
      # Under the root directory of this repo
      # Use -h to see the full usage
      python -m data.compute_dataset_stat_hdf5
      ```

3. Start fine-tuning:

   Configurations relevant to model architecture and data processing are in [this file](configs/base.yaml). Normally, you do not need to modify these configurations; otherwise, it will cause errors in loading the pre-training checkpoint. Configurations relevant to training are passed through *Command Line Arguments*. Use `python main.py -h ` to see the descriptions. We provide an example of a fine-tuning script in [this file](finetune.sh) (`finetune.sh`). You may need to modify some of the parameters in this file, such as `CUTLASS_PATH` and `WANDB_PROJECT`.

   Use this to start fine-tuning:

   ```bash
   source finetune.sh
   ```

   with `finetune.sh` detailed as below:

   ```bash
      deepspeed --hostfile=hostfile.txt main.py \
         --deepspeed="./configs/zero2.json" \   # If you want to use DeepSpeed, which is strongly recommended
         --pretrained_model_name_or_path=<MODEL ID | DIRECTORY OF MODEL WEIGHTS | PATH TO MODEL CHECKPOINT> \
         --pretrained_text_encoder_name_or_path=<MODEL ID | PATH TO MODEL DIRECTORY > \   # e.g., google/t5-v1_1-xxl
         --pretrained_vision_encoder_name_or_path=<MODEL ID | PATH TO MODEL DIRECTORY> \  # e.g., google/siglip-so400m-patch14-384
         --output_dir=<DIRECTORY to SAVE CHECKPOINTS> \ # e.g., checkpoints/rdt-1b-agilex
         --train_batch_size=32 \
         --sample_batch_size=64 \   # batch size for diffusion sampling in validation 
         --max_train_steps=200000 \
         --checkpointing_period=1000 \
         --sample_period=500 \   # sample period for validation
         --checkpoints_total_limit=40 \
         --lr_scheduler="constant" \
         --learning_rate=1e-4 \
         --mixed_precision="bf16" \ # If you want to use mixed precision, bf16 is recommended
         --dataloader_num_workers=8 \
         --image_aug \  # If you want to use image augmentation
         --dataset_type="finetune" \
         --state_noise_snr=40 \  # If you want to add noise to the state
         --load_from_hdf5 \   # If you use HDF5 to store your data
         --report_to=wandb
   ```

   **IMPORTANT**: If you have already chosen to precompute the language embeddings, please specify `--precomp_lang_embed` in the `finetune.sh`.

   Note 1: `pretrained_model_name_or_path` can one of:

      - a string, the *model id* of a pre-trained model hosted inside a model repo on HuggingFace. Please fill with `"robotics-diffusion-transformer/rdt-1b"`, which is the officially-released [RDT-1B model](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)🤗 at HuggingFace. (recommended)
      - a string, the path to a *directory* containing the manually downloaded model weights from HuggingFace, e.g., `"/path/to/rdt-1b"`. You should first manually download the `rdt-1b` directory from this [link](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)🤗.
      - a string, the path to a *directory* containing model weights saved using [`~RDTRunner.save_pretrained`] method. This can be either:
        -  `"checkpoints/rdt-pretrain-1b/checkpoint-<STEP NUMBER>"`: This is the path to the checkpoint saved in the `<STEP NUMBE>` iteration during pre-training. Refer to [this file](docs/pretrain.md) for a tutorial on how to start your own pre-training.
        - `"checkpoints/rdt-pretrain-1b"`: If the pre-training completes normally without any exception, you can specify this path to load the last checkpoint.
      - a string, the path to model checkpoint (`*.pt`) saved by DeepSpeed, e.g., `"checkpoints/rdt-pretrain-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt"` (verified)
      - `None` if you want to randomly initialize the model using configuration at `config_path`.

   Note 2: You can monitor the training process by observing `loss` (through a long window moving average) and `overall_avg_sample_mse` in [Wandb](https://wandb.ai/site) or [TensorBoard](https://www.tensorflow.org/tensorboard). We empirically found that the lower the `overall_avg_sample_mse`, the better the model performs. Usually, fine-tuning is over when this value converges.

   Note 3: If the training oscillates, you can increase the batch size by adding more GPUs or setting a larger `--gradient_accumulation_steps`.

## Deployment on Real-Robots

We have encapsulated the inference of the model into a class named `RoboticDiffusionTransformerModel` (see [this file](scripts/agilex_model.py#L38)). You can call this class's `step()` method for inference. However, you may need to re-implement some parts according to your specific robot. You should at least modify the `_format_joint_to_state()` (L164) and `_unformat_action_to_joint()` (L196) to convert between robot raw actions and unified action vectors that RDT accepts. You may also specify the control frequency of your robot (L49).

**IMPORTANT**: When you feed the images into `step()`, remember the order MUST be `[ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, ext_{t}, right_wrist_{t}, left_wrist_{t}]`.

We provide an example hardware code in [this file](scripts/agilex_inference.py) for deployment on Mobile ALOHA, and the corresponding running script in [this file](inference.sh) (`inference.sh`), which is detailed below;

   ```bash
      python -m scripts.agilex_inference \
         --use_actions_interpolation \
         --pretrained_model_name_or_path=<PATH TO MODEL CHECKPOINT> \  # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt, the same before
         --lang_embeddings_path=<PATH TO YOUR INSTURCTION EMBEDDINGS> \ # e.g. outs/lang_embeddings/your_instr.pt"
         --ctrl_freq=25    # your control frequency
   ```

**IMPORTANT**: If you on-board GPU memory is not enough to encode the language, please refer to [this file](scripts/encode_lang.py) for precomputation and specify the language embedding path in `inference.sh`. Detail instructions are provided below:

   1. Set Required Parameters in `scripts/encode_lang.py`

      ```python
      # ...

      GPU = 0
      MODEL_PATH = "google/t5-v1_1-xxl"
      CONFIG_PATH = "configs/base.yaml"
      SAVE_DIR = "outs/"   # output directory

      # Modify this to your task name and instruction
      TASK_NAME = "handover_pan"
      INSTRUCTION = "Pick up the black marker on the right and put it into the packaging box on the left."

      # Note: if your GPU VRAM is less than 24GB, 
      # it is recommanded to enable offloading by specifying an offload directory. 
      OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

      # ...
      ```

   2. Run the scipt
      ```
      python -m scripts.encode_lang
      ```

Note: If you want to deploy on the Mobile ALOHA robot, don't forget to install the hardware prerequisites (see [this repo](https://github.com/MarkFzp/mobile-aloha)).

## Citation

If you find our work helpful, please cite us:

```bibtex
@article{liu2024rdt,
  title={RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation},
  author={Liu, Songming and Wu, Lingxuan and Li, Bangguo and Tan, Hengkai and Chen, Huayu and Wang, Zhengyi and Xu, Ke and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2410.07864},
  year={2024}
}
```

Thank you!

## License

All the code, model weights, and data are licensed under [MIT license](./LICENSE).
