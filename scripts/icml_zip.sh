#!/usr/bin/env bash

ACTUAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${ACTUAL_DIR}


zip -r icml_files.zip  \
        artificial_new.csv  \
        cocob \
        configs/icml_paper \
        datasets.py \
        distributions.py \
        Dockerfile \
        models.py \
        plot_linear.py \
        preprocess.py \
        requirements.txt \
        scinol/*py \
        scripts/docker_build_n_run.sh \
        icml_reproduce.sh \
        short_names.py \
        test.py \
        util_plot.py
