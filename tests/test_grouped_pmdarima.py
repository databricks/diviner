import pytest
from tests import data_generator
from pmdarima.arima.auto import AutoARIMA
from pmdarima.arima.arima import ARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import FourierFeaturizer
from diviner import GroupedPmdarima
from diviner.utils.pmdarima_utils import (
    _PMDARIMA_MODEL_METRICS,
    _COMPOUND_KEYS,
)

PMDARIMA_MODEL_PARAMS = {
    "order",
    "seasonal_order",
    "start_params",
    "method",
    "maxiter",
    "out_of_sample_size",
    "scoring",
    "trend",
    "with_intercept",
}
SERIES_TEST_COUNT = 2
PARAMS_COLS = [
    item for item in PMDARIMA_MODEL_PARAMS if item not in _COMPOUND_KEYS.keys()
] + ["p", "d", "q", "P", "D", "Q", "s"]


@pytest.fixture(scope="module")
def data():
    test_data = data_generator.generate_test_data(
        column_count=2,
        series_count=SERIES_TEST_COUNT,
        series_size=365 * 6,
        start_dt="2020-01-01",
        days_period=1,
    )
    return test_data


@pytest.fixture(scope="module")
def basic_pmdarima(data):

    arima = AutoARIMA(out_of_sample_size=30)
    return GroupedPmdarima("y", "ds", arima).fit(data.df, data.key_columns)


@pytest.fixture(scope="module")
def basic_arima(data):
    arima = ARIMA(out_of_sample_size=30, order=(2, 0, 3))
    return GroupedPmdarima("y", "ds", arima).fit(data.df, data.key_columns)


@pytest.fixture(scope="module")
def basic_pipeline(data):
    pipeline = Pipeline(
        steps=[
            ("fourier", FourierFeaturizer(k=3, m=7)),
            ("arima", AutoARIMA(out_of_sample_size=60)),
        ]
    )
    return GroupedPmdarima("y", "ds", pipeline).fit(data.df, data.key_columns)


@pytest.fixture(scope="module")
def grouped_obj():
    return GroupedPmdarima("y", "ds", ARIMA(order=(1, 1, 1)))


def test_default_arima_metric_extract(basic_arima):

    metrics = basic_arima.get_metrics()
    metric_cols = metrics.columns
    assert len(metrics) == SERIES_TEST_COUNT
    for item in _PMDARIMA_MODEL_METRICS:
        assert item in metric_cols


def test_default_auto_arima_metric_extract(basic_pmdarima):

    metrics = basic_pmdarima.get_metrics()
    metric_cols = metrics.columns
    assert len(metrics) == SERIES_TEST_COUNT
    for item in _PMDARIMA_MODEL_METRICS:
        assert item in metric_cols


def test_default_pipeline_metric_extract(basic_pipeline):

    metrics = basic_pipeline.get_metrics()
    metric_cols = metrics.columns
    assert len(metrics) == SERIES_TEST_COUNT
    for item in _PMDARIMA_MODEL_METRICS:
        assert item in metric_cols


def test_default_arima_params_extract(basic_arima):

    params = basic_arima.get_model_params()
    param_cols = params.columns

    assert len(params) == SERIES_TEST_COUNT
    for column in PARAMS_COLS:
        assert column in param_cols


def test_default_auto_arima_params_extract(basic_pmdarima):

    params = basic_pmdarima.get_model_params()
    param_cols = params.columns

    assert len(params) == SERIES_TEST_COUNT
    for column in PARAMS_COLS:
        assert column in param_cols


def test_default_pipeline_params_extract(basic_pipeline):

    params = basic_pipeline.get_model_params()
    param_cols = params.columns

    assert len(params) == SERIES_TEST_COUNT
    for column in PARAMS_COLS:
        assert column in param_cols


def test_default_pipeline_forecast_no_conf_int(basic_pipeline):
    forecast_cnt = 10
    forecast_columns = {"forecast", "grouping_key_columns", "key1", "key0"}
    forecast = basic_pipeline.predict(forecast_cnt)

    assert set(forecast.columns).issubset(forecast_columns)
    assert len(forecast) == forecast_cnt * SERIES_TEST_COUNT


def test_default_auto_arima_predict_conf_int(basic_pmdarima):
    forecast_cnt = 10
    forecast_columns = {
        "forecast",
        "grouping_key_columns",
        "key1",
        "key0",
        "yhat_lower",
        "yhat_upper",
    }

    prediction = basic_pmdarima.predict(n_periods=forecast_cnt, return_conf_int=True)

    assert len(prediction) == forecast_cnt * SERIES_TEST_COUNT
    assert set(prediction.columns).issubset(forecast_columns)


@pytest.mark.parametrize("type_", ["additive", "multiplicative"])
def test_group_trend_decomposition(data, grouped_obj, type_):

    decomposed = grouped_obj.decompose_groups(
        data.df, data.key_columns, m=4, type_=type_
    )
    for col in {
        "x",
        "trend",
        "seasonal",
        "random",
        "ds",
        "key1",
        "key0",
        "grouping_key_columns",
    }:
        assert col in decomposed.columns
    assert len(decomposed) == len(data.df)


def test_group_ndfiff_calculation(data, grouped_obj):

    # Purposefully select an alpha that would drive a 'd' value very high
    # (this value should never be used in practice)
    ndiffs = grouped_obj.calculate_ndiffs(
        data.df, data.key_columns, alpha=0.5, test="pp", max_d=2
    )

    assert len(ndiffs) == SERIES_TEST_COUNT
    for k, v in ndiffs.items():
        assert isinstance(k, tuple)
        assert v <= 2


def test_group_nsdiff_calculation(data, grouped_obj):

    nsdiffs = grouped_obj.calculate_nsdiffs(
        data.df, data.key_columns, m=365, test="ch", max_D=5
    )

    assert len(nsdiffs) == SERIES_TEST_COUNT
    for k, v in nsdiffs.items():
        assert isinstance(k, tuple)
        assert v == 4  # Works with the algorithm used to generate the test data


def test_group_is_constant_calculation(data, grouped_obj):

    is_constants_check = grouped_obj.calculate_is_constant(data.df, data.key_columns)

    assert len(is_constants_check) == SERIES_TEST_COUNT
    for k, v in is_constants_check.items():
        assert isinstance(k, tuple)
        assert (
            not v
        )  # The algorithm that generates the data intentionally creates a trend
