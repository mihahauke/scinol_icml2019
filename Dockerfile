FROM ubuntu:16.04

FROM nvidia/cuda:9.0-cudnn7-devel

# Python3 with pip3
RUN pip3 install pip --upgrade


RUN pip3 --no-cache-dir install \
         tensorflow==1.12.0 \
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



