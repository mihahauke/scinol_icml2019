#!/usr/bin/env bash

LOGDIR="tb_logs"
if [ $# -eq 1 ]
  then
    LOGDIR=$1
fi
tensorboard --logdir=${LOGDIR}