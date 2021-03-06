#!/usr/bin/env bash

set -xe

function run(){

    DIR=$(dirname $0)
    BASE=${DIR}/../..
    SBIN=${BASE}/sbin
    source ${SBIN}/env.sh
    cd ${BASE}

    bash ${SBIN}/package.sh

    export PYTHONPATH=$PYTHONPATH:${BASE}/py/dist/fmnist-0.1.0-py3.7.egg

    now=`date +%s`
    tmp_dir_template="fme_${now}-XXXXXXX"
    TMP_DIR=$(mktemp -d -t "${tmp_dir_template}")
    python -m fmnist.learning.arch.fcnn.train \
        --train-data "${HOME}/code/fashion-mnist/data" \
        --job-dir "${TMP_DIR}/job_dir" \
        --model-dir "${TMP_DIR}/model_dir" \
        --num-threads 4 \
        --num-layers 4 \
        --layer-size 512 \
        --batch-size 64 \
        --num-epochs 50 \
        --activation "tanh" \
        --dropout-rate 0.05 \
        --optimizer "nadam" \
        --lr 0.0006 "$@"
}

run "$@"
