#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${ACTUAL_DIR}
NAME="optim_`hostname`"
IMAGE_TAG="optim"


./docker/slurm.sh "mnist" test.py -c connfig/exp/exp_mnist.yml
./docker/slurm.sh "bank" test.py -c connfig/exp/exp_bank.yml
./docker/slurm.sh "census" test.py -c connfig/exp/exp_census.yml
./docker/slurm.sh "covtype" test.py -c connfig/exp/exp_covtype.yml
./docker/slurm.sh "madelon" test.py -c connfig/exp/exp_madelon.yml
./docker/slurm.sh "shuttle" test.py -c connfig/exp/exp_shuttle.yml