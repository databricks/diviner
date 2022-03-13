#!/usr/bin/env bash

set -ex

diviner_envs=$(
  conda env list |                 # list (env name, env path) pairs
  cut -d' ' -f1 |                  # extract env names
  grep "^diviner-[a-z0-9]\{40\}\$"  # filter envs created by diviner
) || true

if [ -n "$diviner_envs" ]; then
  for env in $diviner_envs
  do
    "conda remove --all --yes --name $env"
  done
fi

conda clean --all --yes
conda env list

set +ex
