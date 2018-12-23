#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}/..

# Don't add parentheses, '~' in there won't be resolved correctly
docker_script=~/optimizers/docker/build_n_run.sh
configs_dir="configs/exp_linear"

./docker/slurm.sh "wololo" ${docker_script} ./test.py -c ${configs_dir}/exp_prescinol.yml
