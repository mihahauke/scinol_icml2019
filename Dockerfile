FROM ubuntu:16.04

# Cuda 8 with cudnn 5
FROM nvidia/cuda:9.0-cudnn7-devel

# ViZdoom dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3 \
    python3-pip \
    language-pack-en-base



# Python3 with pip3
RUN pip3 install pip --upgrade


RUN pip3 --no-cache-dir install \
         tensorflow-gpu==1.9.0 \
         ruamel.yaml \
         numpy \
         tqdm \
         python-mnist==0.5 \
         scikit-learn \
         pmlb


RUN echo "export LANG=en_US.UTF-8" > /etc/bash.bashrc

WORKDIR /home/optimizers
ENV HOME /home
RUN chmod 777 /home



