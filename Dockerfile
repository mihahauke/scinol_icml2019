FROM ubuntu:16.04

FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        python3-dev \
        python3 \
        python3-pip \
        language-pack-en-base \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Python3 with pip3
RUN pip3 install pip --upgrade


RUN pip3 --no-cache-dir install \
         setuptools\
         tensorflow-gpu==1.12.0 \
         ruamel.yaml \
         numpy \
         tqdm \
         python-mnist==0.5 \
         scikit-learn==0.20.2\
         pmlb


RUN echo "export LANG=en_US.UTF-8" > /etc/bash.bashrc

WORKDIR /home/optimizers
ENV HOME /home
RUN chmod 777 /home



