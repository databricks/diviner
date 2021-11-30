import inspect
import importlib
import warnings
import pytest
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tests.v1 import data_generator
from diviner.v1.grouped_statsmodels import GroupedStatsmodels
from diviner.v1.exceptions import DivinerException


LAGS_MODELS = {"ARDL", "AutoReg"}
K_FACTOR_MODELS = {"DynamicFactor", "DynamicFactorMQ"}
MARKOV_MODELS = {"MarkovRegression", "MarkovAutoregression"}
VAR_MODELS = {"VAR"}


def get_statsmodels_classes():
    # These models either don't have a fit/predict/forecast API OR throw unconstrained
    # solution exceptions (requiring very specific data sets that are very complex to generate)
    # or have initialization non-determinism that create flaky test cases (~ 10% of the time
    # they will throw an exception due to ill-fitting or convergence problems) with the test
    # data set.
    model_exclusion_set = [
        "AR",
        "ArmaProcess",
        "MarkovAutoregression",
        "STL",
        "STLForecast",
        "SVAR",
        "VECM",
        "UECM",
        "VARMAX",
    ]
    return [
        model
        for model, _ in inspect.getmembers(
            importlib.import_module("statsmodels.tsa.api"), inspect.isclass
        )
        if model not in model_exclusion_set
    ]


def _get_individual_model(model, index):

    _model_key = list(model.model.keys())[index]
    return model.model[_model_key]


def assemble_prediction_df(train_data, start, end):
    grouped = train_data.df.groupby(train_data.key_columns)
    groups = set(grouped.groups.keys())
    prediction_configuration = []
    for keys in groups:
        row = {}
        for index, col in enumerate(train_data.key_columns):
            row[col] = keys[index]
        row["start"] = start
        row["end"] = end
        prediction_configuration.append(row)
    return pd.DataFrame.from_records(prediction_configuration)


@pytest.mark.parametrize(
    "method",
    [
        ("get_model_params", None),
        ("get_metrics", None),
        ("save", "/"),
        ("predict", "placeholder"),
        ("forecast", 1),
    ],
)
def test_statsmodels_fit_checks(method):

    model = GroupedStatsmodels(model_type="AutoReg", endog="y", time_col="ds")

    with pytest.raises(DivinerException, match="The model has not been fit."):
        if method[1]:
            getattr(model, method[0])(method[1])
        else:
            getattr(model, method[0])()


@pytest.mark.parametrize("method", [("fit", ["placeholder", ("a", "b")])])
def test_statsmodels_init_checks(method):
    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)
    model = GroupedStatsmodels(model_type="Holt", endog="y", time_col="ds").fit(
        train.df, train.key_columns
    )

    with pytest.raises(DivinerException, match="The model has already been fit."):
        getattr(model, method[0])(*method[1])


def test_simple_arima_fit():

    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)

    model = GroupedStatsmodels(model_type="ARIMA", endog="y", time_col="ds").fit(
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
            model_type="sarimax", endog="y", exog_column="exog", time_col="ds"
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
        model_type="ExponentialSmoothing", endog="y", time_col="ds"
    ).fit(train.df, train.key_columns, smoothing_seasonal=0.8, remove_bias=True)

    last_model = _get_individual_model(model, -1)

    assert last_model.params["smoothing_seasonal"] == 0.8

    prediction = last_model.predict(start="2020-02-02", end="2020-03-04")

    assert len(prediction) == 8
    assert {row > 0.0 for row in prediction}  # make sure the predictions are positive


@pytest.mark.filterwarnings("ignore")
def test_ets_model_class_and_fit_single_model_with_custom_kwargs():

    train = data_generator.generate_test_data(3, 6, 2000, "2020-01-01", 1)

    model = GroupedStatsmodels(model_type="ETSModel", endog="y", time_col="ds").fit(
        train.df, train.key_columns, maxiter=20, disp=False, trend="mul", error="add"
    )

    last_model = _get_individual_model(model, -1)

    assert last_model.error == "add"
    assert last_model.trend == "mul"

    prediction = last_model.predict(start=1950, end=2100)

    assert len(prediction) == 151
    assert {row > 0.0 for row in prediction}


def test_holt_model_fit_and_metrics_gathering():

    train = data_generator.generate_test_data(3, 6, 2000, "2020-01-01", 1)

    warnings.simplefilter("always")
    model = GroupedStatsmodels(model_type="holt", endog="y", time_col="ds").fit(
        train.df,
        train.key_columns,
        damped_trend=True,
        initialization_method="known",
        initial_level=0.6,
        initial_trend=150.0,
    )

    with warnings.catch_warnings(record=True) as w:
        scores = model.get_metrics(metrics=["mse", "sse", "aic"], warning=True)

        assert len(w) == 6
        assert issubclass(w[-1].category, RuntimeWarning)
        assert len(scores) == 6
        assert {"sse", "aic"}.issubset(set(scores.columns))

    with warnings.catch_warnings(record=True) as w2:
        scores_warnings_to_logs = model.get_metrics(
            metrics=["mse", "sse", "aic"], warning=False
        )
        assert not w2
        assert len(scores_warnings_to_logs) == 6


def test_holt_model_fit_and_grouped_forecast():

    train = data_generator.generate_test_data(3, 6, 2000, "2020-01-01", 1)

    model = GroupedStatsmodels(model_type="holt", endog="y", time_col="ds").fit(
        train.df, train.key_columns, damped_trend=False
    )

    forecast = model.forecast(3)
    assert len(forecast) == 18
    assert {row > 0 for row in forecast["forecast"]}


def test_autoreg_model_fit_and_predict():
    group_count = 8
    train = data_generator.generate_test_data(3, group_count, 2000, "2020-02-02", 1)
    model = GroupedStatsmodels(model_type="autoreg", endog="y", time_col="ds").fit(
        train.df, train.key_columns, lags=12
    )

    prediction_df = assemble_prediction_df(train, "2022-01-01", "2023-02-02")

    prediction = model.predict(prediction_df)

    assert len(prediction) == 398 * group_count
    assert {row > 0 for row in prediction["forecast"]}


@pytest.mark.parametrize("model_type", get_statsmodels_classes())
def test_statsmodels_model_types(model_type):

    group_count = 4
    train = data_generator.generate_test_data(3, group_count, 1000, "2020-02-02", 1)
    train.df["multivar"] = train.df.apply(lambda row: row.y * 0.2, axis=1)

    if model_type in LAGS_MODELS:
        model = GroupedStatsmodels(model_type=model_type, endog="y", time_col="ds").fit(
            train.df, train.key_columns, lags=1
        )
    elif model_type in K_FACTOR_MODELS:
        model = GroupedStatsmodels(
            model_type=model_type, endog=["y", "multivar"], time_col="ds"
        ).fit(
            train.df,
            train.key_columns,
            k_factors=1,
            factor_order=2,
            error_order=2,
            method="powell",
            enforce_stationarity=False,
        )
    elif model_type in MARKOV_MODELS:
        model = GroupedStatsmodels(model_type=model_type, endog="y", time_col="ds").fit(
            train.df, train.key_columns, k_regimes=2, order=4
        )
    elif model_type in VAR_MODELS:
        model = GroupedStatsmodels(
            model_type=model_type, endog=["y", "multivar"], time_col="ds"
        ).fit(train.df, train.key_columns)
    else:
        model = GroupedStatsmodels(model_type=model_type, endog="y", time_col="ds").fit(
            train.df, train.key_columns
        )

    prediction_df = assemble_prediction_df(train, "2021-01-01", "2021-02-02")

    prediction = model.predict(prediction_df)

    assert len(prediction) > 0


def test_statsmodels_param_extract():
    group_count = 8
    train = data_generator.generate_test_data(3, group_count, 2000, "2020-02-02", 1)
    model = GroupedStatsmodels(model_type="Holt", endog="y", time_col="ds").fit(
        train.df, train.key_columns
    )
    params = model.get_model_params()

    assert len(params) == group_count
    assert {row > 0 for row in params["smoothing_level"]}
