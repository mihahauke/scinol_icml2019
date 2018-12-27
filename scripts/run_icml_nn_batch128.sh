#!/usr/bin/env bash

set -x
ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}/..

# Don't add parentheses, '~' in there won't be resolved correctly
docker_script=~/optimizers/scripts/docker_build_n_run.sh
configs_dir="configs/exp_b128_nn"

./scripts/slurm_sbatch.sh "mnist128nn" ${docker_script} ./test.py -c ${configs_dir}/"mnist.yml"
./scripts/slurm_sbatch.sh "bank128nn" ${docker_script} ./test.py -c ${configs_dir}/"bank.yml"
./scripts/slurm_sbatch.sh "census128nn" ${docker_script} ./test.py -c ${configs_dir}/"census.yml"
./scripts/slurm_sbatch.sh "covtype128nn" ${docker_script} ./test.py -c ${configs_dir}/"covtype.yml"
./scripts/slurm_sbatch.sh "madelon128nn" ${docker_script} ./test.py -c ${configs_dir}/"madelon.yml"
./scripts/slurm_sbatch.sh "shuttle128nn" ${docker_script} ./test.py -c ${configs_dir}/"shuttle.yml"
