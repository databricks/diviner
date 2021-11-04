from tests import data_generator
from diviner.grouped_prophet import GroupedProphet
from prophet import Prophet
from datetime import timedelta
import os
import shutil
import pytest


def _get_individual_model(model, index):

    _model_key = list(model.model.keys())[index]
    return model.model[_model_key]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_prophet_default_fit():

    train = data_generator.generate_test_data(4, 2, 1000, "2020-01-01", 1)
    model = GroupedProphet().fit(train.df, train.key_columns)
    first_model = _get_individual_model(model, 0)

    assert len(first_model.history) > 0
    assert (
        len(first_model.params["trend"][0]) == 1000
    )  # fit value for each value in series
    assert len(list(model.model.keys())) == 2


def test_prophet_forecast_correct_start():

    train = data_generator.generate_test_data(2, 5, 1000, "2020-01-01", 1)
    expected_start_of_forecast = max(train.df["ds"]) + timedelta(days=1)
    model = GroupedProphet().fit(train.df, train.key_columns)
    forecasted_data = model.forecast(10, "D")

    # check that the first date in the forecasted df for the first model is 1 day after last date.
    min_forecast = min(forecasted_data["ds"])

    assert expected_start_of_forecast == min_forecast
    assert len(forecasted_data) == 50


def test_prophet_save_and_load():
    # Tests serialization, deserialization, and utilization of forecasting API from loaded model
    save_path = os.path.join("/tmp/grouped_prophet_test", "model")

    train = data_generator.generate_test_data(2, 2, 1000, "2020-01-01", 1)
    grouped_model = GroupedProphet().fit(train.df, train.key_columns)
    grouped_model.save(save_path)
    loaded_model = GroupedProphet().load(save_path)
    forecasts = loaded_model.forecast(25, "D")

    shutil.rmtree(os.path.dirname(save_path))

    assert len(forecasts) == 50


def test_prophet_execution_with_kwargs_override_for_pystan():

    train = data_generator.generate_test_data(4, 6, 1000, "2020-01-01", 1)

    default_prophet_uncertainty_samples = Prophet().uncertainty_samples

    model = GroupedProphet(uncertainty_samples=0).fit(
        train.df, train.key_columns, algorithm="LBFGS"
    )

    last_model = _get_individual_model(model, 5)

    assert last_model.uncertainty_samples == 0
    assert default_prophet_uncertainty_samples != last_model.uncertainty_samples


def test_prophet_cross_validation_extract():

    train = data_generator.generate_test_data(4, 6, 1000, "2020-01-01", 1)

    model = GroupedProphet(uncertainty_samples=0).fit(train.df, train.key_columns)

    scores = model.cross_validation(
        initial="100 days", period="90 days", horizon="15 days", parallel=None
    )

    assert all(scores["rmse"] > 0)
    assert len(scores) == 6
    assert "coverage" not in scores


def test_prophet_cross_validation_extract_custom_scores():

    train = data_generator.generate_test_data(4, 2, 1000, "2020-01-01", 1)

    model = GroupedProphet(uncertainty_samples=0).fit(train.df, train.key_columns)

    scores = model.cross_validation(
        initial="100 days",
        period="90 days",
        horizon="15 days",
        parallel=None,
        metrics=["rmse", "mape"],
        disable_tqdm=False,
    )

    assert all(scores["rmse"] > 0)
    assert len(scores) == 2
    assert "coverage" not in scores
