#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${ACTUAL_DIR}

TAG="$1"
SCRIPT="$2"
ARGS="${@:3}"

LOGDIR="~/slurm_logs"
LOGFILE=${LOGDIR}/`hostname`_${TAG}_`date +"%d_%H_%M_%S"`.log

sbatch  -J ${TAG} \
        --exclusive \
        -p lab-ci \
	    -o ${LOGFILE} \
	    ${SCRIPT} ${ARGS}


