#!/usr/bin/env bash
set -x
err=0
trap 'err=1' ERR
DIVINER_HOME=$(pwd)
export DIVINER_HOME

pytest "$DIVINER_HOME"/tests/examples/test_examples.py

test $err = 0
