#!/bin/bash
#SBATCH -o flashattn.%j.out
#SBATCH -e flashattn.%j.err
#SBATCH --partition=i64m1tga800u
#SBATCH -J install_flashattn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # You get 16 threads for compiling
#SBATCH --gres=gpu:1
#SBATCH --qos=low

# Activate your conda environment first
source ~/.bashrc
conda activate rdt  # or your specific conda env name

module load cuda/12.2

python -c "import torch; print(torch.cuda.device_count())"
nvcc -V

pip install --upgrade pip wheel setuptools

# Install flash-attn
pip install flash-attn --no-build-isolation
