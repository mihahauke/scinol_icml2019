#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${ACTUAL_DIR}

TAG="$1"
SCRIPT="$2"
ARGS="${@:3}"

LOGDIR="/home/mkempka/slurm_logs"
LOGFILE=${LOGDIR}/`date +"%m_%d_%H_%M_%S"`_`hostname`_${TAG}.log

sbatch  -J ${TAG} \
	-p lab-ci \
	--exclusive \
	-t 4-00:00:00 \
	-x lab-ci-5 \
	-o ${LOGFILE} \
	 ${SCRIPT} ${ARGS}


