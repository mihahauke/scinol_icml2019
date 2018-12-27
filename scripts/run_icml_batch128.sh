#!/usr/bin/env bash

set -x
ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}/..

# Don't add parentheses, '~' in there won't be resolved correctly
docker_script=~/optimizers/docker/build_n_run.sh
configs_dir="configs/exp_b128"

./docker/slurm.sh "mnist128" ${docker_script} ./test.py -c ${configs_dir}/"mnist.yml"
./docker/slurm.sh "bank128" ${docker_script} ./test.py -c ${configs_dir}/"bank.yml"
./docker/slurm.sh "census128" ${docker_script} ./test.py -c ${configs_dir}/"census.yml"
./docker/slurm.sh "covtype128" ${docker_script} ./test.py -c ${configs_dir}/"covtype.yml"
./docker/slurm.sh "madelon128" ${docker_script} ./test.py -c ${configs_dir}/"madelon.yml"
./docker/slurm.sh "shuttle128" ${docker_script} ./test.py -c ${configs_dir}/"shuttle.yml"
