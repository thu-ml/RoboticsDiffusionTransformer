## 用kinova数据finetune
### 处理数据
```shell
    #scripts/process_hdf5.py
    
    #将原始的image放入cam_high,cam_left_wrist,cam_right_wrist，根据需要选择将数据填充指定位置相机，其余相机做0填充
    #训练数据中的图像数据应为bgr格式，若为rgb格式，则用process_rgb_image函数转换，否则直接用process_bgr_image函数
    #替换该函数中调用的函数名，以及process_image中处理的数据名(image1)
    #目前使用双相机(image1为外部 image2为右臂)，左腕做0填充，需要配置并执行两次该函数分别处理image1和image2
    process_images_in_parallel()

    #将原eef数据(6维)处理为eef6d(9维)，结果为xyz+6d+gripper共10维
    hdf5_eef6d_process()

    #替换原action为使用eef6d
    hdf5_action_process()
```

### 数据配置
```shell
    #删除多余数据集的配置信息
    configs/dataset_control_freq.json
    configs/finetune_datasets.json
    configs/finetune_sample_weights.json
```

### 训练配置
```shell
    #根据实际修改finetune.sh以下配置
    export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
    export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
    export CUTLASS_PATH="/path/to/cutlass"
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b"

    #配置data/hdf5_vla_dataset.py下的数据路径
    HDF5_DIR = "data/datasets/agilex/rdt_data/"
    self.DATASET_NAME = "agilex"

    #若使用单卡训练配置finetune.sh
    accelerate launch main.py \
    #多卡则使用
    deepspeed --hostfile=hostfile.txt main.py \
```
### 训练
```shell
    #文本预编码，修改TARGET_DIR，之后修改data/hdf5_vla_dataset中instruction路径
    python scripts/encode_lang_batch.py

    #计算数据集统计信息
    python -m data.compute_dataset_stat_hdf5

    #启动
    source finetune.sh
```

## 推理并执行
```shell
#假设已经按着作者的要求把rdt-1b,t5,siglip下载后并且链接上
#当前单臂实现对应是右手(right-arm)
cd RoboticsDiffusionTransformer
mkdir lang_outs
python -m scripts.encode_lang
#注意修改`pretrained_model_name_or_path`, `lang_embeddings_path`
#如果用kinova自带末端相机, `img_right_topic`不需要改,否则需要改
# ！注意inference_fn函数中根据输入图像是rgb或bgr做相应的处理
python -m scripts.kinova_inference

```