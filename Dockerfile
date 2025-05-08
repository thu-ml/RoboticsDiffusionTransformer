# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set the maintainer label
LABEL maintainer="heng.zhang@iit.it" \
      version="0.1" \
      description="Robotics Diffusion Transformer with CUDA 12.4.1 support"

# Set environment variables
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHON_VERSION=3.10 \
    TZ=Europe/Rome

# Install system dependencies
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3-pip \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies in correct order
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir "numpy<2"  # Force NumPy 1.x version

# Install PyTorch with CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install packaging and other dependencies
RUN pip3 install --no-cache-dir \
    packaging==24.0 \
    transformers \
    diffusers

# Install flash-attn with build constraints
RUN pip3 install --no-cache-dir flash-attn==2.5.7 --no-build-isolation

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set default command
CMD ["python3", "-u", "main.py"]
