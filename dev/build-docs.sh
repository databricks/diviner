#!/usr/bin/env bash
set -x
err=0
trap 'err=1' ERR
DIVINER_HOME=$(pwd)
export DIVINER_HOME

(cd docs && make clean)
(cd docs && make html SPHINXOPTS="-W --keep-going -n")

test $err = 0