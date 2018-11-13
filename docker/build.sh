#!/usr/bin/env bash

cd ~/optimizers

docker build -t ${TAG} .
./run.sh $@
