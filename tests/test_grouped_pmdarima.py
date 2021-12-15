import pytest
from tests import data_generator
from pmdarima.arima.auto import AutoARIMA
from pmdarima.arima.arima import ARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import FourierFeaturizer
from diviner import GroupedPmdarima
from diviner.utils.pmdarima_utils import (
    _PMDARIMA_MODEL_METRICS,
    _PMDARIMA_MODEL_PARAMS,
    _COMPOUND_KEYS,
    generate_prediction_config,
)


SERIES_TEST_COUNT = 2
PARAMS_COLS = [
    item for item in _PMDARIMA_MODEL_PARAMS if item not in _COMPOUND_KEYS.keys()
] + ["p", "d", "q", "P", "D", "Q", "s"]


@pytest.fixture(scope="module")
def data():
    test_data = data_generator.generate_test_data(
        column_count=2,
        series_count=SERIES_TEST_COUNT,
        series_size=2000,
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
def undefined_model(data):
    return GroupedPmdarima("y", "ds").fit(
        data.df, data.key_columns, arima__out_of_sample_size=30
    )


@pytest.mark.parametrize(
    "conf, expecting",
    [
        (AutoARIMA(), ["pmdarima.arima.auto", "AutoARIMA"]),
        (ARIMA(order=(1, 0, 2)), ["pmdarima.arima.arima", "ARIMA"]),
        (Pipeline(steps=[("arima", AutoARIMA())]), ["pmdarima.pipeline", "Pipeline"]),
    ],
    ids=["AutoARIMA", "ARIMA", "Pipeline"],
)
def test_configuration(conf, expecting):
    grouped_model = GroupedPmdarima("y", "ds", conf)
    assert grouped_model._module == expecting[0]
    assert grouped_model._clazz == expecting[1]


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


def test_default_generate_pipeline_metric_extract(undefined_model):
    metrics = undefined_model.get_metrics()
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


def test_default_generate_pipeline_params_extract(undefined_model):
    params = undefined_model.get_model_params()
    param_cols = params.columns

    assert len(params) == SERIES_TEST_COUNT
    for column in PARAMS_COLS:
        assert column in param_cols


def test_default_pipeline_forecast_no_conf_int(basic_pipeline):
    forecast_cnt = 10
    forecast_columns = {"forecast", "grouping_key_columns", "key1", "key0"}
    forecast = basic_pipeline.forecast(forecast_cnt)

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

    prediction_conf = generate_prediction_config(
        list(basic_pmdarima.model.keys()),
        basic_pmdarima._group_key_columns,
        n_periods=forecast_cnt,
        return_conf_int=True,
    )

    prediction = basic_pmdarima.predict(prediction_conf)

    assert len(prediction) == forecast_cnt * SERIES_TEST_COUNT
    assert set(prediction.columns).issubset(forecast_columns)
