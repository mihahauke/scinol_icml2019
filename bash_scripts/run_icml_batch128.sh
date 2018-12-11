#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}/..

docker_script="~/optimizers/docker/build_n_run.sh"
configs_dir="configs/exp_icml_batch128"

./docker/slurm.sh "mnist" ${docker_script} ./test.py -c ${configs_dir}/exp_mnist.yml
./docker/slurm.sh "bank" ${docker_script} ./test.py -c ${configs_dir}/exp_bank.yml
./docker/slurm.sh "census" ${docker_script} ./test.py -c ${configs_dir}/exp_census.yml
./docker/slurm.sh "covtype" ${docker_script} ./test.py -c ${configs_dir}/exp_covtype.yml
./docker/slurm.sh "madelon" ${docker_script} ./test.py -c ${configs_dir}/exp_madelon.yml
./docker/slurm.sh "shuttle" ${docker_script} ./test.py -c ${configs_dir}/exp_shuttle.yml
