import os
from tests.v1 import data_generator
from diviner.v1.grouped_statsmodels import GroupedStatsmodels


_EXCLUDE_TEST_ATTR = {"model"}


def test_statsmodels_save_and_load():

    path = "/tmp/modeltest_holt"
    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)
    model = GroupedStatsmodels(model_type="Holt", endog=["y"], time_col="ds").fit(
        train.df, train.key_columns
    )
    model.save(path)
    test_elements = {key for key in vars(model).keys() if key not in _EXCLUDE_TEST_ATTR}
    loaded = GroupedStatsmodels.load(path)
    for attr in test_elements:
        assert getattr(model, attr) == getattr(loaded, attr)
    os.remove(path)


def test_statsmodels_save_and_load_and_predict():

    path = "/tmp/modeltest_arima"
    train = data_generator.generate_test_data(2, 2, 1000, "2020-02-02", 1)
    model = GroupedStatsmodels(model_type="ARIMA", endog=["y"], time_col="ds").fit(
        train.df, train.key_columns
    )
    model.save(path)
    loaded = GroupedStatsmodels.load(path)
    forecast = loaded.forecast(horizon=120, frequency="D")

    assert len(forecast) == 240
    assert {row > 0 for row in forecast["forecast"]}
    os.remove(path)
