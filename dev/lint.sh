#!/usr/bin/env bash

DIVINER_HOME=$(pwd)
export DIVINER_HOME

err=0
trap 'err=1' ERR

echo -e "$(tput bold; tput setaf 6)\n========== black ==========\n$(tput sgr0)"
# Exclude proto files because they are auto-generated

if ! black --check .;
then
  echo "
To apply black foramtting to your PR, run:
``pip install '$(grep "^black==" requirements/lint-requirements.txt)' && black .``
from the repository root."
fi

echo -e "$(tput bold; tput setaf 6)\n========== pylint ==========\n$(tput sgr0)"
pylint $(git ls-files | grep '\.py$') --rcfile=$DIVINER_HOME/pylintrc

echo -e "$(tput bold; tput setaf 6)\n========== rstcheck ==========\n$(tput sgr0)"
rstcheck $(git ls-files | grep '\.rst$')

if [[ "$err" != "0" ]]; then
  echo -e "\n$(tput bold; tput setaf 1)One of the lint checks failed. Check the above stages for detailed failure reasons.$(tput sgr0)"
else
  echo -e "\n$(tput bold; tput setaf 2)All lint checks passed!$(tput sgr0)"
fi

test $err = 0
