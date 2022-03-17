#!/usr/bin/env bash

showHelp() {
cat << EOF
Usage: ./build-dist.sh [-r] [-v]
Script for validating the development environment, building Diviner, validating packages, validating the docs build,
and submitting the built package to pypi.

-h, -help,          --help                  Display help

-r, -release,       --release               Release Diviner to pypi

-v, -verbose,       --verbose               Run script in verbose mode. Will print out each step of execution.

EOF
}


export release=0
export verbose=0
export dev_build=0

while getopts "rvh" f
do
  case "$f" in
    r) release=1 ;;
    v) verbose=1 ;;
    h) showHelp; exit ;;
    *) showHelp; exit ;;
  esac
done

if [[ $verbose = 1 ]]; then
  set -exv
fi

function retry-with-backoff() {
    for BACKOFF in 0 1 2 4 8 16 32 64; do
        sleep $BACKOFF
        if "$@"; then
            return 0
        fi
    done
    return 1
}

python --version
pip install --upgrade pip wheel
pip --version

DIVINER_HOME=$(pwd)
export DIVINER_HOME

# Add requirements
required_files=" -r ./requirements/base-requirements.txt"
required_files+=" -r ./requirements/lint-requirements.txt"
required_files+=" -r ./requirements/docs-requirements.txt"

if [[ -n $required_files ]]; then
  retry-with-backoff pip install $required_files
fi

# ensure that the docs build
"$DIVINER_HOME"/dev/build-docs.sh
test $? = 0


diviner_ver=$(< diviner/version.py grep -o 'VERSION = "[^"]\+"' | sed 's/[^"]*"\([^"]*\).*/\1/')
echo "$(tput bold; tput setaf 2)Diviner version: $(tput sgr0)${diviner_ver}"

dev_warning="$(tput bold; tput setaf 1)Warning! This is a development version!$(tput sgr0)"
export dev_warning

if grep -q "dev" <<< "$diviner_ver"; then
  dev_build=1
  dev_warning
fi

read -p "Do you wish to proceed with building this release version? $(tput bold)(y/n)$(tput sgr0): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
  # build the package
  python "$DIVINER_HOME"/setup.py sdist bdist_wheel
  tar tzf "$DIVINER_HOME"/dist/diviner-${diviner_ver}.tar.gz
  twine check "$DIVINER_HOME"/dist/*

  test $? = 0

  if [[ $release = 1 && $dev_build = 0 ]]; then
    read -p "Do you wish to upload this build to pypi? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      # release to pypi
      echo "$(tput bold; tput setaf 3)Uploading distribution to pypi... $(tput sgr0)"
      # TODO: uncomment when repo is public
      # twine upload "$DIVINER_HOME"/dist/*
      echo "$(tput bold; tput setaf 2)Upload complete!$(tput sgr0)"
      exit
    fi
    elif [[ $release = 1 && $dev_build = 1 ]]; then
      echo "$(tput bold; tput setaf 1)Cannot upload dev version to pypi! Correct the version first.$(tput sgr0)"
    else
      exit
  fi
else
  exit
fi