from tests import data_generator
from diviner import GroupedPmdarima
from diviner.utils.pmdarima_utils import (
    _extract_arima_model_from_pipeline,
    _get_arima_params,
    _get_arima_training_metrics,
    _PMDARIMA_MODEL_METRICS,
)


def test_default_arima_fit_attribute_extraction():

    data = data_generator.generate_test_data(
        column_count=2,
        series_count=2,
        series_size=2000,
        start_dt="2020-01-01",
        days_period=1,
    )

    arima_model = GroupedPmdarima("y", "ds").fit(
        data.df, data.key_columns, arima__out_of_sample_size=30
    )

    for group, model in arima_model.model.items():
        pipeline = arima_model._extract_individual_pipeline(group)
        instance_model = _extract_arima_model_from_pipeline(pipeline)

        group_metrics = _get_arima_training_metrics(instance_model)

        for key, value in group_metrics.items():
            assert value > 0
            assert key in _PMDARIMA_MODEL_METRICS
        for item in _PMDARIMA_MODEL_METRICS:
            assert item in group_metrics.keys()

        group_params = _get_arima_params(instance_model)

        for item in {"P", "D", "Q", "s"}:  # this isn't a seasonality model
            assert group_params[item] == 0
