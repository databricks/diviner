import pytest

from tests import data_generator
from diviner import GroupedStatsmodels
from diviner.v1.exceptions import DivinerException
from diviner.v1.scoring.statsmodels_scoring import (
    _STATSMODELS_METRICS,
    _extract_statsmodels_single_model_metrics,
)


def test_statsmodels_scoring_extracts_valid():

    train = data_generator.generate_test_data(2, 1, 2000, "2020-01-01", 1)
    model = GroupedStatsmodels(model_type="arima", endog=["y"], time_col="ds").fit(
        train.df, train.key_columns
    )
    individual_model = list(model.model.values())[0]
    model_metrics = _extract_statsmodels_single_model_metrics(
        individual_model, _STATSMODELS_METRICS, False
    )
    for metric in _STATSMODELS_METRICS:
        instance_metric = getattr(individual_model, metric)
        assert instance_metric > 0
        assert model_metrics[metric] == instance_metric


def test_statsmodels_scoring_raises_on_invalid_metric():
    train = data_generator.generate_test_data(2, 1, 2000, "2020-01-01", 1)
    model = GroupedStatsmodels(model_type="arima", endog=["y"], time_col="ds").fit(
        train.df, train.key_columns
    )
    with pytest.raises(DivinerException, match=f"Metrics supplied are invalid: "):
        model.get_metrics(metrics={"rmse", "aic", "bic", "invalid"})
