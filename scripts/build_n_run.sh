#!/usr/bin/env bash

ROOT_DIR="/home/mkempka/optimizers"
cd ${ROOT_DIR}

IMAGE_TAG="optim"
docker build -t ${IMAGE_TAG} .
./scripts/run.sh  "$@"
