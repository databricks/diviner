from tests import data_generator
from pmdarima.arima.auto import AutoARIMA
from diviner import GroupedPmdarima
from diviner.utils.pmdarima_utils import (
    _extract_arima_model,
    _get_arima_params,
    _get_arima_training_metrics,
    _PMDARIMA_MODEL_METRICS,
    _construct_prediction_config,
)


def test_default_arima_fit_attribute_extraction():

    data = data_generator.generate_test_data(
        column_count=2,
        series_count=2,
        series_size=2000,
        start_dt="2020-01-01",
        days_period=1,
    )

    arima_model = GroupedPmdarima(
        "y", "ds", model_template=AutoARIMA(out_of_sample_size=30)
    ).fit(data.df, data.key_columns)

    for group, model in arima_model.model.items():
        pipeline = arima_model._extract_individual_model(group)
        instance_model = _extract_arima_model(pipeline)

        group_metrics = _get_arima_training_metrics(instance_model)

        for key, value in group_metrics.items():
            assert value > 0
            assert key in _PMDARIMA_MODEL_METRICS
        for item in _PMDARIMA_MODEL_METRICS:
            assert item in group_metrics.keys()

        group_params = _get_arima_params(instance_model)

        for item in {"P", "D", "Q", "s"}:  # this isn't a seasonality model
            assert group_params[item] == 0


def test_prediction_config_generation():

    group_keys = [("a", "z"), ("b", "z")]

    conf = _construct_prediction_config(
        group_keys=group_keys,
        group_key_columns=["col1", "col2"],
        n_periods=50,
        alpha=0.7,
        return_conf_int=True,
        inverse_transform=False,
    )

    for idx, row in conf.iterrows():
        assert row.get("col1") == group_keys[idx][0]
        assert row.get("col2") == group_keys[idx][1]
        assert row.get("n_periods") == 50
        assert row.get("alpha") == 0.7
        assert row.get("return_conf_int")
        assert not row.get("inverse_transform")
