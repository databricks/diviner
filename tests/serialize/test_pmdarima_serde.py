import pytest
from pandas.testing import assert_frame_equal
from pmdarima.arima.auto import AutoARIMA
from pmdarima.pipeline import Pipeline
from tests import data_generator
from diviner import GroupedPmdarima
from diviner.serialize.pmdarima_serializer import PmdarimaEncoder, PmdarimaDecoder

SERIES_TEST_COUNT = 2


@pytest.fixture(scope="module")
def data():
    test_data = data_generator.generate_test_data(
        column_count=3,
        series_count=SERIES_TEST_COUNT,
        series_size=365 * 5,
        start_dt="2020-01-01",
        days_period=1,
    )
    return test_data


@pytest.fixture(scope="module")
def model(data):

    arima = GroupedPmdarima(
        model_template=Pipeline(steps=[("arima", AutoARIMA(out_of_sample_size=60, max_order=7))]),
    ).fit(
        df=data.df,
        group_key_columns=data.key_columns,
        y_col="y",
        datetime_col="ds",
        silence_warnings=True,
    )
    return arima


def test_grouped_pmdarima_serialize(model):

    model_attrs = vars(model)
    encoded = PmdarimaEncoder().encode(model)
    decoded = PmdarimaDecoder().decode(encoded)

    for key in model_attrs.keys():
        if key not in {"model", "_model_template"}:
            assert model_attrs[key] == decoded[key]
        else:
            assert isinstance(decoded[key], type(model_attrs[key]))


def test_grouped_pmdarima_save_and_load(model):

    orig_params = model.get_model_params()
    orig_metrics = model.get_metrics()
    save_path = "/tmp/pmdarima/test.pmd"
    model.save(save_path)

    loaded = GroupedPmdarima.load(save_path)
    loaded_params = loaded.get_model_params()
    loaded_metrics = loaded.get_metrics()

    assert_frame_equal(orig_params, loaded_params)
    assert_frame_equal(orig_metrics, loaded_metrics)


def test_grouped_pmdarima_save_load_predict(model):

    save_path = "/tmp/pmdarima/test.pmd"
    forecast = model.predict(30, return_conf_int=True)
    model.save(save_path)
    loaded = GroupedPmdarima.load(save_path)
    loaded_forecast = loaded.predict(30, return_conf_int=True)

    assert_frame_equal(forecast, loaded_forecast)
