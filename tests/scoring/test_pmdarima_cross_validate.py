import numpy as np
from diviner import GroupedPmdarima
from diviner.scoring.pmdarima_cross_validate import (
    _cross_validate_single_model,
)
from tests import data_generator
from pmdarima.model_selection import SlidingWindowForecastCV, RollingForecastCV
from pmdarima.arima.auto import AutoARIMA


def test_individual_model_cross_validate():

    metrics = ["smape", "mean_squared_error", "mean_absolute_error"]

    train = data_generator.generate_test_data(1, 1, 765, "2019-01-01")

    model = AutoARIMA(max_order=5, out_of_sample_size=30).fit(train.df["y"])
    cross_validator = SlidingWindowForecastCV(window_size=180, step=120, h=90)

    cv_results = _cross_validate_single_model(
        model,
        train.df["y"],
        metrics=metrics,
        cross_validator=cross_validator,
        error_score=np.nan,
        exog=None,
        verbosity=3,
    )

    expected_fields = [f"{met}_mean" for met in metrics] + [f"{met}_stddev" for met in metrics]
    for key, value in cv_results.items():
        assert key in expected_fields
        assert value > 0
        if "_stddev" in key:
            assert value < cv_results.get(key.split("_stddev")[0] + "_mean") * 10.0


def test_grouped_model_cross_validate():

    metrics = ["smape", "mean_squared_error", "mean_absolute_error"]
    expected_columns = (
        [f"{met}_mean" for met in metrics]
        + [f"{met}_stddev" for met in metrics]
        + ["grouping_key_columns", "key0"]
    )

    train = data_generator.generate_test_data(1, 2, 765, "2019-01-01")

    grouped_model = GroupedPmdarima(
        model_template=AutoARIMA(max_order=5, out_of_sample_size=30),
    ).fit(train.df, train.key_columns, "y", "ds", silence_warnings=True)
    cross_validator = RollingForecastCV(h=90, step=120, initial=365)
    cv_metrics = grouped_model.cross_validate(train.df, metrics, cross_validator)

    assert len(cv_metrics) == 2
    assert set(cv_metrics.columns).issubset(set(expected_columns))
