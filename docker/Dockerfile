# Based on https://github.com/naruya/dl_remote/blob/master/Dockerfile
# [1] https://github.com/robbyrussell/oh-my-zsh
# [2] https://github.com/pyenv/pyenv/wiki/common-build-problems

# FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        pkg-config \
        unzip

# zsh (from [1]) ----------------
RUN apt-get update && apt-get install -y \
    wget git zsh
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN sed -i 's/# DISABLE_AUTO_UPDATE="true"/DISABLE_AUTO_UPDATE="true"/g' ~/.zshrc

# pyenv (from [2]) ----------------
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y \
#     build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
#     libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
#     xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# X window ----------------
# RUN apt-get update && apt-get install -y \
    # xvfb x11vnc python-opengl icewm
# RUN echo 'alias vnc="export DISPLAY=:0; Xvfb :0 -screen 0 1400x900x24 &; x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 &; icewm-session &"' >> /root/.zshrc

# utils ----------------
RUN apt-get update && apt-get install -y \
    vim \
    libsdl2-2.0-0 \
    cmake \ 
    # xvfb \
    xorg-dev \
    freeglut3-dev \
    libglu1-mesa-dev \
    x11-apps

# robomimic dependencies ----------------
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    python3-pip

COPY requirements.txt /root/

# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
RUN pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# Install packaging
RUN pip install packaging==24.0

# Install flash-attn
RUN pip install flash-attn 
#--no-build-isolation

# Install other prequisites
RUN pip install -r /root/requirements.txt



ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# finalize image  ----------------
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN source ~/.zshrc

WORKDIR /root
CMD ["zsh"]
