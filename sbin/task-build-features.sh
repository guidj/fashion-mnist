#!/usr/bin/env bash

set -xe

function run(){

    DIR=$(dirname $0)
    BASE=${DIR}/..
    SBIN=${BASE}/sbin
    source ${SBIN}/env.sh
    cd ${BASE}

    bash ${SBIN}/package.sh

    export now=`date +%s`
    export job_id="fme_${now}"
    gcloud ai-platform jobs submit training ${job_id} \
        --module-name=fmnist.features.build_features \
        --job-dir="gs://${GCS_DATA}/ai-platform/jobs/${job_id}" \
        --labels=motto=fff \
        --package-path=${BASE}/py/fmnist \
        --python-version=3.5 \
        --region=${GCP_REGION} \
        --runtime-version=1.14 \
        --scale-tier=CUSTOM \
        --master-machine-type=large_model_v100 \
        --stream-logs \
        -- \
        --train-data "gs://${GCS_DATA}/data/"
    gsutil -m cp -r "${GCS_DATA}/data/interim/*.npz" ${BASE}/data/interim/
}

run "$@"
