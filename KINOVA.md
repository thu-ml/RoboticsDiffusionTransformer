# RDT-1B: Finetune and Inference on Kinova Robot

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
    **IMPORTANT :** We train the model using the end-effector of the robotic arm.We use [6D representation](https://arxiv.org/pdf/1812.07035) for EEF rotation. You can process your end-effector data into the format of our dataset(see [this script](scripts/convert_rpy.py)).

    
2. Implement the dataset loader:

   You need to:

   1. Register the configuration of `my_cool_dataset`:

      Append the control frequency of `my_cool_dataset` in [this file](configs/dataset_control_freq.json). Write the name of `my_cool_dataset` in [this file](configs/finetune_datasets.json) and [this file](configs/finetune_sample_weights.json), where the value of the sampling weight doesn't matter since you only have one dataset. In these two files, we leave a placeholder of `agilex`; you can simply replace it with `my_cool_dataset`.

   2. Re-Implement the class of `HDF5VLADataset`:

      You can find this class in [this file](data/hdf5_vla_dataset.py). In this file, we provide an example of loading the fine-tuning dataset used in our paper (see [this link](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data)).

      To adapt it to your dataset, you need to: modify the `HDF5_DIR` (directory to `my_cool_dataset`) and `DATASET_NAME` (should be `"my_cool_dataset"`) in L21 and L22; 

      **IMPORTANT :** If you use RTX 4090 (or lower), the GPU memory may be too low to load the `t5-v1_1-xxl` encoder. Instead, we recommend you precompute the language embeddings (see [this file](scripts/encode_lang_batch.py) for an example script) and load them during training. In this way, you need to specify the path to the embeddings in the `HDF5VLADataset` (see L148) rather than the natural language.

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

## Deployment on Kinova Robot

We have encapsulated the inference of the model into a class named `KinovaDiffusionTransformerModel` (see [this file](scripts/kinova_model.py)). You can call this class's `step()` method for inference. 

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
      # it is recommended to enable offloading by specifying an offload directory. 
      OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

      # ...
      ```

   2. Run the script
      ```
      python -m scripts.encode_lang
      ```

We provide hardware code in [this file](scripts/kinova_inference.py) for deployment on Kinova, You need to modify the args [here](scripts/kinova_inference.py#L557) to match your environment.

Run the scripts starting  inference

   ```bash
      python -m scripts.kinova_inference
   ```

Note: If you want to deploy on the Kinova robot, don't forget to install the hardware prerequisites (see [this repo](https://github.com/Kinovarobotics/ros_kortex)).


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
