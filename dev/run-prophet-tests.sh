#!/usr/bin/env bash
set -x
err=0
trap 'err=1' ERR
DIVINER_HOME=$(pwd)
export DIVINER_HOME

pytest tests/test_grouped_prophet.py
pytest tests/config/test_prophet_config.py
pytest tests/scoring/test_prophet_cross_validate.py
pytest tests/serialize/test_prophet_serde.py

test $err = 0
