#!/usr/bin/env bash

err=0
trap 'err=1' ERR

echo -e "\n========== black ==========\n"
# Exclude proto files because they are auto-generated

if ! black --check .;
then
  echo "
To apply black foramtting to your PR, run:
``pip install '$(grep "^black==" requirements/lint-requirements.txt)' && black .``
from the repository root."
fi

echo -e "\n========== pylint ==========\n"
pylint $(git ls-files | grep '\.py$')

echo -e "\n========== rstcheck ==========\n"
rstcheck $(git ls-files | grep '\.rst$')

if [[ "$err" != "0" ]]; then
  echo -e "\nOne of the lint checks failed. Check the above stages for detailed failure reasons."
fi

test $err = 0
