#!/usr/bin/env bash

set -x
ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}/..

# Don't add parentheses, '~' in there won't be resolved correctly
docker_script=~/optimizers/scripts/docker_build_n_run.sh
configs_dir="configs/exp_linear_add"

./scripts/slurm_sbatch.sh "amnist" ${docker_script} ./test.py -c ${configs_dir}/exp_mnist.yml
./scripts/slurm_sbatch.sh "abank" ${docker_script} ./test.py -c ${configs_dir}/exp_bank.yml
./scripts/slurm_sbatch.sh "acensus" ${docker_script} ./test.py -c ${configs_dir}/exp_census.yml
./scripts/slurm_sbatch.sh "acovtype" ${docker_script} ./test.py -c ${configs_dir}/exp_covtype.yml
./scripts/slurm_sbatch.sh "amadelon" ${docker_script} ./test.py -c ${configs_dir}/exp_madelon.yml
./scripts/slurm_sbatch.sh "ashuttle" ${docker_script} ./test.py -c ${configs_dir}/exp_shuttle.yml


