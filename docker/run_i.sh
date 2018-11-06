#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${ACTUAL_DIR}

NAME="optim_`hostname`"
IMAGE_TAG="optim"

nvidia-docker run \
       --user=`id -u`:`id -g`\
       --net=host \
       -it \
       --name ${NAME} \
       -v ${ACTUAL_DIR}:/home/optimizers \
       --rm\
       --entrypoint /bin/bash \
        ${IMAGE_TAG}
