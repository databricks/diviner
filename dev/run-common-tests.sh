#!/usr/bin/env bash
set -x
err=0
trap 'err=1' ERR
DIVINER_HOME=$(pwd)
export DIVINER_HOME

pytest tests/data
pytest tests/utils

test $err = 0
