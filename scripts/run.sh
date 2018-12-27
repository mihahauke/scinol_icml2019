#!/usr/bin/env bash

ROOT_DIR="/home/mkempka/optimizers"
cd ${ROOT_DIR}

NAME="optim_$RANDOM"
IMAGE_TAG="optim"

nvidia-docker run \
        --user=`id -u`:`id -g`\
        --name ${NAME} \
        -v ${ROOT_DIR}:/home/optimizers \
        --rm\
        ${IMAGE_TAG} \
        "$@"
