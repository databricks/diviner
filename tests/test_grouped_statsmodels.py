from tests import data_generator
from diviner.grouped_statsmodels import GroupedStatsmodels
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pytest
import warnings

def _get_individual_model(model, index):
    #  TODO: abstract this into a test helper module

    _model_key = list(model.model.keys())[index]
    return model.model[_model_key]


def test_simple_arima_fit():

    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)

    model = GroupedStatsmodels(model_type="ARIMA", endog_column="y", time_col="ds").fit(
        train.df, train.key_columns, order=(2, 1, 1), trend="n"
    )

    first_model = _get_individual_model(model, 0)

    assert first_model.specification["trend"] == "n"
    assert first_model.specification["order"] == (2, 1, 1)


@pytest.mark.filterwarnings("ignore")
def test_exogeneous_regressor_fit():

    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)
    train.df["exog"] = train.df.apply(lambda row: row.y * 0.2, axis=1)

    with pytest.warns(ConvergenceWarning):
        model = GroupedStatsmodels(
            model_type="sarimax", endog_column="y", exog_column="exog", time_col="ds"
        ).fit(
            train.df,
            train.key_columns,
            order=(1, 0, 1),
            seasonal_order=(0, 0, 0, 4),
            enforce_stationarity=False,
            enforce_invertibility=False,
            disp=False,  # turn off the LBFGS stdout warning for convergence
        )

        first_model = _get_individual_model(model, 0)

        assert first_model.specification["order"] == (1, 0, 1)
        assert not first_model.specification["enforce_stationarity"]
        assert len(model.model.keys()) == 2


@pytest.mark.filterwarnings("ignore")
def test_holt_winters_fit_and_predict_single_model():

    train = data_generator.generate_test_data(3, 4, 1000, "2020-01-01", 4)

    model = GroupedStatsmodels(
        model_type="ExponentialSmoothing", endog_column="y", time_col="ds"
    ).fit(train.df, train.key_columns, smoothing_seasonal=0.8, remove_bias=True)

    last_model = _get_individual_model(model, -1)

    assert last_model.params["smoothing_seasonal"] == 0.8

    prediction = last_model.predict(start="2020-02-02", end="2020-03-04")

    assert len(prediction) == 8
    assert {row > 0.0 for row in prediction}  # make sure the predictions are positive


@pytest.mark.filterwarnings("ignore")
def test_ets_model_class_and_fit_single_model_with_custom_kwargs():

    train = data_generator.generate_test_data(3, 6, 2000, "2020-01-01", 1)

    model = GroupedStatsmodels(
        model_type="ETSModel", endog_column="y", time_col="ds"
    ).fit(train.df, train.key_columns, maxiter=20, disp=False, trend="mul", error="add")

    last_model = _get_individual_model(model, -1)

    assert last_model.error == "add"
    assert last_model.trend == "mul"

    prediction = last_model.predict(start=1950, end=2100)

    assert len(prediction) == 151
    assert {row > 0.0 for row in prediction}


def test_holt_model_fit_and_metrics_gathering():

    train = data_generator.generate_test_data(3, 6, 2000, "2020-01-01", 1)

    warnings.simplefilter("always")
    model = GroupedStatsmodels(
        model_type="holt", endog_column="y", time_col="ds"
    ).fit(train.df, train.key_columns, damped_trend=True,
          initialization_method="known", initial_level=0.6, initial_trend=150.0)

    with warnings.catch_warnings(record=True) as w:
        scores = model.score_model(metrics=['mse', 'sse', 'aic'], warning=True)

        assert len(w) == 6
        assert issubclass(w[-1].category, RuntimeWarning)
        assert len(scores) == 6
        assert {"sse", "aic"}.issubset(set(scores.columns))

    with warnings.catch_warnings(record=True) as w2:
        scores_warnings_to_logs = model.score_model(metrics=['mse', 'sse', 'aic'], warning=False)
        assert not w2
        assert len(scores_warnings_to_logs) == 6


def test_holt_model_fit_and_grouped_forecast():

    train = data_generator.generate_test_data(3, 6, 2000, "2020-01-01", 1)

    model = GroupedStatsmodels(
        model_type="holt", endog_column="y", time_col="ds"
    ).fit(train.df, train.key_columns, damped_trend=False)

    forecast = model.forecast(3)

    print(forecast)

    # assert 1 == 0
    #TODO: clean this up