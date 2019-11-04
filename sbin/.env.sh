#!/usr/bin/env bash

set -xe

export GCP_PROJECT='my-project'
export GCS_BUCKET='my-bucket'
export GCS_DATA="${GCS_BUCKET}/path-to-my-data"
export GCP_REGION=some-region
