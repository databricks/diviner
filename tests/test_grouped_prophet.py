import pandas as pd

from tests import data_generator
from diviner import GroupedProphet
from diviner.exceptions import DivinerException
from prophet import Prophet
from datetime import timedelta, datetime
import os
import shutil
import pytest


def _get_individual_model(model, index):

    _model_key = list(model.model.keys())[index]
    return model.model[_model_key]


def test_model_raises_if_already_fit():
    train = data_generator.generate_test_data(2, 1, 1000, "2020-01-01", 1)
    model = GroupedProphet().fit(train.df, train.key_columns)
    with pytest.raises(
        DivinerException,
        match="The model has already been fit. Create a new instance to fit the model again.",
    ):
        model.fit(train.df, train.key_columns)


def test_model_raises_if_not_fit():
    model = GroupedProphet()
    with pytest.raises(
        DivinerException,
        match="The model has not been fit. Please fit the model first.",
    ):
        model.forecast(30, "days")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_prophet_default_fit():

    train = data_generator.generate_test_data(4, 2, 1000, "2020-01-01", 1)
    model = GroupedProphet().fit(train.df, train.key_columns)
    first_model = _get_individual_model(model, 0)

    assert len(first_model.history) > 0
    assert len(first_model.params["trend"][0]) == 1000  # fit value for each value in series
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
    loaded_model = GroupedProphet.load(save_path)
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

    scores = model.cross_validate_and_score(
        initial="100 days", period="90 days", horizon="15 days", parallel=None
    )

    assert all(scores["rmse"] > 0)
    assert len(scores) == 6
    assert "coverage" not in scores


def test_prophet_cross_validation_extract_custom_scores():

    train = data_generator.generate_test_data(4, 2, 1000, "2020-01-01", 1)

    model = GroupedProphet(uncertainty_samples=0).fit(train.df, train.key_columns)

    scores = model.cross_validate_and_score(
        initial="100 days",
        period="90 days",
        horizon="15 days",
        parallel=None,
        metrics=["rmse", "mape"],
        disable_tqdm=False,
        monthly=True,
    )

    assert all(scores["rmse"] > 0)
    assert len(scores) == 2
    assert "coverage" not in scores


def test_prophet_extract_params():
    train = data_generator.generate_test_data(4, 6, 1000, "2020-01-01", 1)

    model = GroupedProphet(uncertainty_samples=0).fit(train.df, train.key_columns)

    params = model.extract_model_params()

    assert len(params) == 6


def test_prophet_with_bad_group_data():

    train = data_generator.generate_test_data(2, 1, 1000, "2020-01-01", 1)
    train_df = train.df
    bad_data = pd.DataFrame(
        {
            "ds": datetime.strptime("2021-01-01", "%Y-%M-%d"),
            "y": -500.3,
            "key1": "bad",
            "key0": "data",
        },
        index=[1000],
    )

    train_df_add = pd.concat([train_df, bad_data])

    with pytest.warns(RuntimeWarning, match="An error occurred while fitting group"):
        model = GroupedProphet().fit(train_df_add, train.key_columns)
    assert ("bad", "data") not in model.model.keys()


def test_prophet_df_naming_overrides():

    train = data_generator.generate_test_data(2, 1, 1000, "2020-01-01", 1)
    train_df = train.df
    train_df.rename(columns={"ds": "datetime", "y": "sales"}, inplace=True)

    assert {"datetime", "sales"}.issubset(set(train_df.columns))

    model = GroupedProphet().fit(train_df, train.key_columns, "sales", "datetime")

    params = model.extract_model_params()

    assert len(params) == 1


def test_prophet_manual_predict():
    train = data_generator.generate_test_data(2, 1, 1000, "2020-01-01", 1)
    train_df = train.df

    predict_df = train_df[["key1", "key0", "ds"]][-10:]

    model = GroupedProphet().fit(train_df, train.key_columns)

    prediction = model.predict(predict_df)

    assert len(prediction) == 10

    for _, row in prediction.iterrows():
        assert row["yhat"] > 0


def test_prophet_group_subset_predict():

    _rows_to_generate = 30
    train = data_generator.generate_test_data(2, 1, 1000, "2020-01-01", 1)
    train_df = train.df

    model = GroupedProphet().fit(train_df, train.key_columns)

    key_entries = []
    for v in train_df[["key1", "key0"]].iloc[[0]].to_dict().values():
        key_entries.append(list(v.values())[0])
    groups = [tuple(key_entries)]

    group_prediction = model.predict_groups(groups, _rows_to_generate, "D")

    assert len(group_prediction) == _rows_to_generate
    _key1 = group_prediction["key1"].unique()
    assert len(_key1) == 1
    assert _key1[0] == groups[0][0]
    _key0 = group_prediction["key0"].unique()
    assert len(_key0) == 1
    assert _key0[0] == groups[0][1]


def test_prophet_group_subset_predict_raises_and_warns():

    _rows_to_generate = 30
    train = data_generator.generate_test_data(2, 1, 1000, "2020-01-01", 1)
    train_df = train.df

    model = GroupedProphet().fit(train_df, train.key_columns)

    key_entries = []
    for v in train_df[["key1", "key0"]].iloc[[0]].to_dict().values():
        key_entries.append(list(v.values())[0])
    groups = [(key_entries[0], key_entries[1]), ("missing", "key")]

    with pytest.raises(DivinerException, match="Cannot perform predictions due to submitted"):
        model.predict_groups(groups, _rows_to_generate, "D")

    with pytest.warns(UserWarning, match="Specified groups are unable to be predicted due to "):
        model.predict_groups(groups, _rows_to_generate, "D", on_error="warn")

    with pytest.raises(DivinerException, match="Groups specified for subset forecasting are not"):
        model.predict_groups(("invalid", "invalid"), _rows_to_generate, "D", on_error="ignore")
