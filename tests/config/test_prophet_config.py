from prophet import Prophet
from diviner.config.grouped_prophet.prophet_config import (
    get_scoring_metrics,
    _get_extract_params,
)
from diviner.config.grouped_prophet.utils import prophet_config_utils


def test_base_cv_metrics_extract():

    init_prophet = Prophet()

    base = get_scoring_metrics()
    metric_result_positive_uncertainty = prophet_config_utils._reconcile_metrics(
        base, init_prophet.uncertainty_samples
    )

    assert "coverage" in metric_result_positive_uncertainty

    setattr(init_prophet, "uncertainty_samples", 0)

    metric_result_no_uncertainty = prophet_config_utils._reconcile_metrics(
        base, init_prophet.uncertainty_samples
    )

    assert "coverage" not in metric_result_no_uncertainty
    assert "coverage" in base  # validate immutability for iterable traversal purposes


def test_param_extract():

    full_prophet_params = dir(Prophet())

    assert set(_get_extract_params()).issubset(set(full_prophet_params))
