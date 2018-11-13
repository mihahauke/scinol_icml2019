#!/usr/bin/env bash

cd ~/optimizers
IMAGE_TAG="optim"
docker build -t ${IMAGE_TAG} .
./docker/run.sh  "$@"
