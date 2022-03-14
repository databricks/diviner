#!/usr/bin/env bash
set -x

DIVINER_HOME=$(pwd)
export DIVINER_HOME

clean_docs() {
  cd docs || return
  make clean
  return $?
}

make_docs() {
  make html SPHINXOPTS="-W --keep-going -n"
  return $?
}
clean_docs
clean=$?

make_docs
res=$?

cd "$DIVINER_HOME" || exit

err=$((clean + res))

test $err = 0