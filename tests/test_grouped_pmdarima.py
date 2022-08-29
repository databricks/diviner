import pytest
from tests import data_generator
from pmdarima.arima.auto import AutoARIMA
from pmdarima.arima.arima import ARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import LogEndogTransformer
from diviner import GroupedPmdarima, PmdarimaAnalyzer
from diviner.exceptions import DivinerException
from diviner.utils.pmdarima_utils import _COMPOUND_KEYS, _PMDARIMA_MODEL_METRICS

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
PARAMS_COLS = [item for item in PMDARIMA_MODEL_PARAMS if item not in _COMPOUND_KEYS.keys()] + [
    "p",
    "d",
    "q",
    "P",
    "D",
    "Q",
    "s",
]


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
    return GroupedPmdarima(arima).fit(
        data.df,
        data.key_columns,
        "y",
        "ds",
    )


@pytest.fixture(scope="module")
def basic_arima(data):
    arima = ARIMA(out_of_sample_size=30, order=(2, 0, 3))
    return GroupedPmdarima(arima).fit(
        data.df,
        data.key_columns,
        "y",
        "ds",
    )


@pytest.fixture(scope="module")
def basic_pipeline(data):
    pipeline = Pipeline(
        steps=[
            ("logendog", LogEndogTransformer(neg_action="warn", floor=1.0)),
            ("arima", AutoARIMA(out_of_sample_size=60)),
        ]
    )
    return GroupedPmdarima(pipeline).fit(
        data.df,
        data.key_columns,
        "y",
        "ds",
    )


@pytest.fixture(scope="module")
def pipeline_override_d(data):
    pipeline = Pipeline(steps=[("arima", AutoARIMA(out_of_sample_size=30))])
    util = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    )
    ndiffs = util.calculate_ndiffs(alpha=0.2, test="kpss", max_d=7)
    nsdiffs = util.calculate_nsdiffs(m=7, test="ocsb", max_D=7)
    return GroupedPmdarima(pipeline).fit(
        df=data.df,
        group_key_columns=data.key_columns,
        y_col="y",
        datetime_col="ds",
        ndiffs=ndiffs,
        nsdiffs=nsdiffs,
        silence_warnings=True,
    )


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
    forecast_columns = {"forecast", "grouping_key_columns", "key1", "key0", "ds"}
    forecast = basic_pipeline.predict(forecast_cnt, "forecast")

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
        "ds",
    }

    prediction = basic_pmdarima.predict(
        n_periods=forecast_cnt, predict_col="forecast", return_conf_int=True
    )

    assert len(prediction) == forecast_cnt * SERIES_TEST_COUNT
    assert set(prediction.columns).issubset(forecast_columns)


def test_pmdarima_stationarity_optimized_overrides(data, pipeline_override_d):

    ndiffs = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_ndiffs(alpha=0.5, test="kpss", max_d=7)

    params = pipeline_override_d.get_model_params()

    for _, row in params.iterrows():
        group = (row["key1"], row["key0"])
        assert ndiffs.get(group) == row["d"]
        assert (
            row["D"] == 0
        )  # this isn't a seasonal model so the override shouldn't populate for 'D'


def test_pmdarima_group_subset_predict(data, basic_pmdarima):

    forecast_rows = 30

    train_df = data.df

    key_entries = []
    for v in train_df[["key1", "key0"]].iloc[[0]].to_dict().values():
        key_entries.append(list(v.values())[0])
    groups = [tuple(key_entries)]

    group_prediction = basic_pmdarima.predict_groups(groups, forecast_rows)
    assert len(group_prediction) == forecast_rows
    _key1 = group_prediction["key1"].unique()
    assert len(_key1) == 1
    assert _key1[0] == groups[0][0]
    _key0 = group_prediction["key0"].unique()
    assert len(_key0) == 1
    assert _key0[0] == groups[0][1]


def test_pmdarima_group_subset_predict_pipeline(data, basic_pipeline):

    forecast_rows = 30

    train_df = data.df

    key_entries = []
    for v in train_df[["key1", "key0"]].iloc[[0]].to_dict().values():
        key_entries.append(list(v.values())[0])
    groups = [tuple(key_entries)]

    group_prediction = basic_pipeline.predict_groups(groups, forecast_rows)
    assert len(group_prediction) == forecast_rows
    _key1 = group_prediction["key1"].unique()
    assert len(_key1) == 1
    assert _key1[0] == groups[0][0]
    _key0 = group_prediction["key0"].unique()
    assert len(_key0) == 1
    assert _key0[0] == groups[0][1]


def test_pmdarima_group_subset_predict_raises_and_warns(data, basic_pipeline):

    forecast_rows = 60

    train_df = data.df

    key_entries = []
    for v in train_df[["key1", "key0"]].iloc[[0]].to_dict().values():
        key_entries.append(list(v.values())[0])
    groups = [(key_entries[0], key_entries[1]), ("bogus", "entry")]

    with pytest.raises(DivinerException, match="Cannot perform predictions due to submitted"):
        basic_pipeline.predict_groups(groups, forecast_rows, "D")

    with pytest.warns(UserWarning, match="Specified groups are unable to be predicted due to "):
        basic_pipeline.predict_groups(groups, forecast_rows, "D", on_error="warn")

    with pytest.raises(DivinerException, match="Groups specified for subset forecasting are not"):
        basic_pipeline.predict_groups(("invalid", "invalid"), forecast_rows, "D", on_error="ignore")
