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
    job_id="fme_${now}-XXXXXXX"
    TMP_DIR=$(mktemp -d -t $job_id)
    python -m fmnist.learning.arch.cvnn.train \
        --train-data "${HOME}/code/fashion-mnist/data" \
        --job-dir "${TMP_DIR}/job_dir" \
        --model-dir "${TMP_DIR}/model_dir" \
        --num-threads 4 \
        --num-blocks 4 \
        --block-size 1 \
        --fcl-num-layers 2 \
        --fcl-layer-size 512 \
        --fcl-dropout-rate 0.05 \
        --batch-size 64 \
        --num-epochs 10 \
        --activation "elu" \
        --optimizer "nadam" \
        --lr 0.0006 "$@"
}

run "$@"
