#!/usr/bin/env bash

ROOT_DIR="/home/mkempka/optimizers"
cd ${ROOT_DIR}

NAME="optim_`hostname`"
IMAGE_TAG="optim"

nvidia-docker run \
       --user=`id -u`:`id -g`\
       --net=host \
       -it \
       --name ${NAME} \
       -v ${ROOT_DIR}:/home/optimizers \
       --rm\
       --entrypoint /bin/bash \
        ${IMAGE_TAG}
