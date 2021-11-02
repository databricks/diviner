from prophet import Prophet
from diviner.config.grouped_prophet.prophet_config import get_extract_params, get_base_metrics


def test_base_cv_metrics_extract():

    init_prophet = Prophet()

    metric_result_positive_uncertainty = get_base_metrics(getattr(init_prophet, "uncertainty_samples"))

    assert "coverage" in metric_result_positive_uncertainty

    setattr(init_prophet, "uncertainty_samples", 0)

    metric_result_no_uncertainty = get_base_metrics(getattr(init_prophet, "uncertainty_samples"))

    assert "coverage" not in metric_result_no_uncertainty


def test_param_extract():

    full_prophet_params = dir(Prophet())

    assert set(get_extract_params()).issubset(set(full_prophet_params))
