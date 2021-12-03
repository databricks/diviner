from diviner.grouped_prophet import GroupedProphet
from diviner.scoring import prophet_cross_validate
from tests import data_generator
import pytest


def test_group_cross_validation():

    train = data_generator.generate_test_data(2, 4, 1000, "2020-01-01", 1)
    model = GroupedProphet(n_changepoints=10, uncertainty_samples=0).fit(
        train.df, train.key_columns
    )

    cv_results = prophet_cross_validate.group_cross_validation(
        model,
        horizon="30 days",
        period="120 days",
        initial="180 days",
        parallel="threads",
    )

    first_key = list(cv_results.keys())[0]
    first_result = cv_results[first_key]

    assert len(set(cv_results.keys())) == 4
    assert {"yhat", "y", "ds", "cutoff"}.issubset(set(first_result.columns))


def test_group_performance_metrics():

    train = data_generator.generate_test_data(2, 4, 1000, "2020-01-01", 1)
    model = GroupedProphet(n_changepoints=10, uncertainty_samples=0).fit(
        train.df, train.key_columns
    )

    cv_results = prophet_cross_validate.group_cross_validation(
        model,
        horizon="30 days",
        period="120 days",
        initial="180 days",
        parallel="threads",
    )

    with pytest.raises(ValueError):
        bad_metrics = ["rmse", "mse", "invalid"]
        prophet_cross_validate.group_performance_metrics(
            cv_results, model, bad_metrics, rolling_window=0.25
        )
    metrics = ["rmse", "mse", "mape"]
    metric_results = prophet_cross_validate.group_performance_metrics(
        cv_results, model, metrics, rolling_window=0.05, monthly=False
    )

    first_result = metric_results[list(metric_results.keys())[0]]

    assert len(set(metric_results.keys())) == 4
    assert set(metrics).issubset(set(first_result.columns))


def test_backtesting_cross_validation():
    train = data_generator.generate_test_data(2, 4, 1000, "2020-01-01", 1)
    model = GroupedProphet(n_changepoints=10, uncertainty_samples=0).fit(
        train.df, train.key_columns
    )

    cv_results = model.cross_validate(
        horizon="30 days", period="180 days", initial="365 days", parallel="processes"
    )
    for entries in cv_results.values():
        assert len(entries) == 120
        assert {row > 0 for row in entries["yhat"]}


def test_manual_performance_metrics_execution():
    train = data_generator.generate_test_data(2, 4, 1000, "2020-01-01", 1)
    model = GroupedProphet(n_changepoints=10, uncertainty_samples=0).fit(
        train.df, train.key_columns
    )

    cv_results = model.cross_validate(
        horizon="30 days", period="180 days", initial="365 days", parallel="processes"
    )
    performance_metrics = model.calculate_performance_metrics(cv_results=cv_results)

    assert len(performance_metrics.keys()) == 4

    metrics = {"mse", "rmse", "mae", "mape", "mdape", "smape"}
    for entries in performance_metrics.values():
        for metric in metrics:
            assert {row > 0 for row in entries[metric]}
