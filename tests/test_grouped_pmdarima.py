import pytest
from tests import data_generator
from diviner import GroupedPmdarima
from diviner.utils.pmdarima_utils import (
    _PMDARIMA_MODEL_METRICS,
    _PMDARIMA_MODEL_PARAMS,
    _COMPOUND_KEYS,
)


@pytest.fixture(scope="module")
def basic_pmdarima():
    data = data_generator.generate_test_data(
        column_count=2,
        series_count=2,
        series_size=2000,
        start_dt="2020-01-01",
        days_period=1,
    )

    return GroupedPmdarima("y", "ds").fit(
        data.df, data.key_columns, arima__out_of_sample_size=30
    )


def test_default_arima_training_metric_extract(basic_pmdarima):

    metrics = basic_pmdarima.get_metrics()
    metric_cols = metrics.columns
    assert len(metrics) == 2
    for item in _PMDARIMA_MODEL_METRICS:
        assert item in metric_cols


def test_default_arima_tracking_params_extract(basic_pmdarima):

    params = basic_pmdarima.get_model_params()
    param_cols = params.columns

    column_checks = [
        item for item in _PMDARIMA_MODEL_PARAMS if item not in _COMPOUND_KEYS.keys()
    ]
    full_checks = column_checks + ["p", "d", "q", "P", "D", "Q", "s"]

    assert len(params) == 2
    for column in full_checks:
        assert column in param_cols
