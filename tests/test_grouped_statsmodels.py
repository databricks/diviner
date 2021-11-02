from tests import data_generator
from diviner.grouped_statsmodels import GroupedStatsmodels


def _get_individual_model(model, index):
    #  TODO: abstract this into a test helper module

    _model_key = list(model.model.keys())[index]
    return model.model[_model_key]


def test_simple_arima_fit():

    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)

    model = GroupedStatsmodels(model_type="ARIMA", endog_column="y").fit(
        train.df, train.key_columns, order=(2, 1, 1), trend="n"
    )

    first_model = _get_individual_model(model, 0)

    assert first_model.specification["trend"] == "n"
    assert first_model.specification["order"] == (2, 1, 1)


def test_exogeneous_regressor_fit():

    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)
    train.df["exog"] = train.df.apply(lambda row: row.y * 0.2, axis=1)

    model = GroupedStatsmodels(
        model_type="sarimax", endog_column="y", exog_column="exog"
    ).fit(
        train.df,
        train.key_columns,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    first_model = _get_individual_model(model, 0)

    assert first_model.specification["order"] == (1, 0, 1)
    assert not first_model.specification["enforce_stationarity"]
    assert len(model.model.keys()) == 2


#  TODO: suppress the pytest warnings for SARIMAX unconstrained fit.

