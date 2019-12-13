#!/usr/bin/env bash

set -xe

export GCP_PROJECT='my-project'
export GCS_BUCKET='my-bucket'
export GCS_DATA="${GCS_BUCKET}/path-to-my-data"
export GCP_REGION=some-region

# tensorflow
# prevent up-front full gpu-allocation (which can lead to memory issues)
export TF_FORCE_GPU_ALLOW_GROWTH=true
