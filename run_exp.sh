#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}
NAME="optim_`hostname`"
IMAGE_TAG="optim"

docker_script="docker/build_n_run.sh"

./docker/slurm.sh "mnist" ${docker_script} ./test.py -c connfig/exp/exp_mnist.yml
./docker/slurm.sh "bank" ${docker_script} ./test.py -c connfig/exp/exp_bank.yml
./docker/slurm.sh "census" ${docker_script} ./test.py -c connfig/exp/exp_census.yml
./docker/slurm.sh "covtype" ${docker_script} ./test.py -c connfig/exp/exp_covtype.yml
./docker/slurm.sh "madelon" ${docker_script} ./test.py -c connfig/exp/exp_madelon.yml
./docker/slurm.sh "shuttle" ${docker_script} ./test.py -c connfig/exp/exp_shuttle.yml