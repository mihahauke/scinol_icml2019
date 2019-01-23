#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${ACTUAL_DIR}

for i in `seq 1 10`; do
    ./test.py -c configs/icml_paper/bank.yml
    ./test.py -c configs/icml_paper/census.yml
    ./test.py -c configs/icml_paper/covtype.yml
    ./test.py -c configs/icml_paper/madelon.yml
    ./test.py -c configs/icml_paper/mnist.yml
    ./test.py -c configs/icml_paper/shuttle.yml
done

./plot_linear.py --key cross_entropy
./plot_linear.py --key accuracy