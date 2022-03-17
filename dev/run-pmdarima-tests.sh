#!/usr/bin/env bash
set -x
err=0
trap 'err=1' ERR
DIVINER_HOME=$(pwd)
export DIVINER_HOME

pytest tests/test_grouped_pmdarima.py
pytest tests/analysis/test_pmdarima_analyzer.py
pytest tests/scoring/test_pmdarima_cross_validate.py
pytest tests/serialize/test_pmdarima_serde.py

test $err = 0
