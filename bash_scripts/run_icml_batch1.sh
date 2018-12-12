#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}/..

# Don't add parentheses, '~' in there won't be resolved correctly
docker_script=~/optimizers/docker/build_n_run.sh
configs_dir="configs/exp_icml_batch1"

./docker/slurm.sh "mnist" ${docker_script} ./test.py -c ${configs_dir}/"mnist.yml"
./docker/slurm.sh "bank" ${docker_script} ./test.py -c ${configs_dir}/"bank.yml"
./docker/slurm.sh "census" ${docker_script} ./test.py -c ${configs_dir}/"census.yml"
./docker/slurm.sh "covtype" ${docker_script} ./test.py -c ${configs_dir}/"covtype.yml"
./docker/slurm.sh "madelon" ${docker_script} ./test.py -c ${configs_dir}/"madelon.yml"
./docker/slurm.sh "shuttle" ${docker_script} ./test.py -c ${configs_dir}/"shuttle.yml"
./docker/slurm.sh "artificial" ${docker_script} ./test.py -c ${configs_dir}/"art.yml"
