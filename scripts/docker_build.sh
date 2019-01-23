#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${ACTUAL_DIR}

IMAGE_TAG="optim"
docker build -t ${IMAGE_TAG} .
