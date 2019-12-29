#!/usr/bin/env bash

set -xe

function run(){

    DIR=$(dirname $0)
    source ${DIR}/env.sh

    cd ${DIR}/../py

    rm -rf build/ dist/
    python setup.py bdist_egg
}

run "$@"
