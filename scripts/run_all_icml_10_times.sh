#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${ACTUAL_DIR}/..


echo "Running icml experiments 10 times from: " `pwd`

for i in ` seq 1 10`;
do
    ./scripts/run_icml_batch128.sh
    ./scripts/run_icml_nn_batch128.sh
    ./scripts/run_linear.sh
done











