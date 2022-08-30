from numpy.testing import assert_allclose
import pytest
from tests import data_generator
from pmdarima.arima.auto import AutoARIMA
from diviner import GroupedPmdarima, PmdarimaAnalyzer
from diviner.exceptions import DivinerException
from diviner.utils.pmdarima_utils import (
    _extract_arima_model,
    _get_arima_params,
    _get_arima_training_metrics,
    _construct_prediction_config,
    _PMDARIMA_MODEL_METRICS,
)

SERIES_TEST_COUNT = 2


@pytest.fixture(scope="module")
def data():
    test_data = data_generator.generate_test_data(
        column_count=2,
        series_count=SERIES_TEST_COUNT,
        series_size=365 * 4,
        start_dt="2018-01-01",
        days_period=1,
    )
    return test_data


def test_pmdarima_default_arima_fit_attribute_extraction(data):

    arima_model = GroupedPmdarima(model_template=AutoARIMA(out_of_sample_size=30)).fit(
        data.df, data.key_columns, "y", "ds"
    )

    for group in arima_model.model.keys():
        pipeline = arima_model._extract_individual_model(group)
        instance_model = _extract_arima_model(pipeline)

        group_metrics = _get_arima_training_metrics(instance_model)

        for key, value in group_metrics.items():
            assert value > 0
            assert key in _PMDARIMA_MODEL_METRICS
        for item in _PMDARIMA_MODEL_METRICS:
            assert item in group_metrics.keys()

        group_params = _get_arima_params(instance_model)

        for item in {"P", "D", "Q", "s"}:  # this isn't a seasonality model
            assert group_params[item] == 0


def test_pmdarima_prediction_config_generation():

    group_keys = [("a", "z"), ("b", "z")]

    conf = _construct_prediction_config(
        group_keys=group_keys,
        group_key_columns=["col1", "col2"],
        n_periods=50,
        alpha=0.7,
        return_conf_int=True,
        inverse_transform=False,
    )

    for idx, row in conf.iterrows():
        assert row.get("col1") == group_keys[idx][0]
        assert row.get("col2") == group_keys[idx][1]
        assert row.get("n_periods") == 50
        assert row.get("alpha") == 0.7
        assert row.get("return_conf_int")
        assert not row.get("inverse_transform")


@pytest.mark.parametrize("type_", ["additive", "multiplicative"])
def test_pmdarima_utils_trend_decomposition(data, type_):

    decomposed = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).decompose_groups(m=7, type_=type_)
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


def test_pmdarima_utils_ndiffs_calculation(data):

    ndiffs = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_ndiffs(alpha=0.2, test="kpss", max_d=7)
    assert len(ndiffs) == SERIES_TEST_COUNT
    for k, v in ndiffs.items():
        assert isinstance(k, tuple)
        assert v <= 7


def test_pmdarima_utils_nsdiffs_calculation(data):

    nsdiffs = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_nsdiffs(m=7, test="ocsb", max_D=7)
    assert len(nsdiffs) == SERIES_TEST_COUNT
    for k, v in nsdiffs.items():
        assert isinstance(k, tuple)
        assert v <= 7


def test_pmdarima_constancy_validation(data):

    constancy = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_is_constant()

    assert len(constancy) == SERIES_TEST_COUNT
    for value in constancy.values():
        assert not value


def test_pmdarima_ndiffs_override_class_args(data):

    ndiffs = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_ndiffs(alpha=0.4, max_d=4)

    base_template = AutoARIMA(d=10, out_of_sample_size=7)

    model = GroupedPmdarima(base_template).fit(
        df=data.df,
        group_key_columns=data.key_columns,
        y_col="y",
        datetime_col="ds",
        ndiffs=ndiffs,
        silence_warnings=True,
    )

    params = model.get_model_params()

    for _, row in params.iterrows():
        assert row["d"] <= 4


def test_pmdarima_calculate_acf_full_args(data):

    acf_data = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_acf(unbiased=True, nlags=90, qstat=True, fft=True, alpha=0.1)

    for payload in acf_data.values():
        assert {"acf", "qstat", "pvalues", "confidence_intervals"}.issubset(payload.keys())
        assert len(payload.get("acf")) == 91
        assert len(payload.get("qstat")) == 90
        assert len(payload.get("confidence_intervals")) == 91
        assert len(payload.get("pvalues")) == 90


def test_pmdarima_calculate_acf_minimal_args(data):

    acf_data = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_acf(unbiased=False, nlags=90, qstat=False, fft=False, alpha=None)
    for payload in acf_data.values():
        assert {"acf"}.issubset(payload.keys())
        assert any(
            key not in payload.keys() for key in ["qstat", "pvalues", "confidence_intervals"]
        )
        assert len(payload.get("acf")) == 91


def test_pmdarima_calculate_pacf_full_args(data):

    pacf_data = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_pacf(nlags=90, method="yw", alpha=0.05)

    for payload in pacf_data.values():
        assert {"pacf", "confidence_intervals"}.issubset(payload.keys())
        assert len(payload.get("pacf")) == 91
        assert len(payload.get("confidence_intervals")) == 91


def test_pmdarima_calculate_pacf_minimal_args(data):

    pacf_data = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).calculate_pacf()

    for payload in pacf_data.values():
        assert {"pacf"}.issubset(payload.keys())
        assert any(key not in payload.keys() for key in ["confidence_intervals"])
        assert len(payload.get("pacf")) == 32


def test_pmdarima_generate_diff(data):

    diff = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    ).generate_diff(lag=2, differences=1)

    for data in diff.values():
        assert len(data["diff"]) == (365 * 4) - 2
        assert data["series_start"] > 0
        assert isinstance(data["series_start"], float)


def test_pmdarima_reconstruct_series_from_diff_inv(data):

    analyzer = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    )
    diff = analyzer.generate_diff(lag=2, differences=1)

    group_dfs = analyzer._group_df

    inverted = analyzer.generate_diff_inversion(diff, lag=2, differences=1, recenter=True)

    for group, data in group_dfs:

        assert_allclose(data["y"], inverted.get(group), rtol=0.1)


def test_pmdarima_diff_inv_fails_with_invalid_data(data):

    analyzer = PmdarimaAnalyzer(
        df=data.df, group_key_columns=data.key_columns, y_col="y", datetime_col="ds"
    )
    diff = analyzer.generate_diff(lag=1, differences=1)

    with pytest.raises(DivinerException, match="group_diff_data does not contain the key `diff`"):
        diff_mod = {}
        for key, value in diff.items():
            diff_mod[key] = {"series_start": value.get("series_start")}
        analyzer.generate_diff_inversion(
            group_diff_data=diff_mod, lag=1, differences=1, recenter=True
        )

    with pytest.warns(
        UserWarning, match="Recentering is not possible due to `series_start` missing"
    ):
        diff_mod = {}
        for key, value in diff.items():
            diff_mod[key] = {"diff": value.get("diff")}
        analyzer.generate_diff_inversion(
            group_diff_data=diff_mod, lag=1, differences=1, recenter=True
        )
