#!/usr/bin/env bash

set -xe

function run(){

    DIR=$(dirname $0)
    BASE=${DIR}/..
    SBIN=${BASE}/sbin
    source ${SBIN}/env.sh
    cd ${BASE}

    bash ${SBIN}/package.sh

    export PYTHONPATH=$PYTHONPATH:${BASE}/py/dist/fmnist-0.1.0-py3.7.egg

    export now=`date +%s`
    export job_id="fme_${now}"
    TMP_DIR=$(mktemp -d -t $job_id)
    python -m fmnist.models.train_model \
        --train-data "${HOME}/code/fashion-mnist/data" \
        --model-dir "${TMP_DIR}"
}

run "$@"
