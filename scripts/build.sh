#!/usr/bin/env bash

cd /home/mkempka/optimizers
IMAGE_TAG="optim"
docker build -t ${IMAGE_TAG} .
