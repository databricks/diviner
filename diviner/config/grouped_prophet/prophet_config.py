from prophet import Prophet
from typing import Dict
from copy import deepcopy

BASE_PROPHET_ATTRS = [
    "growth",
    "changepoints",
    "n_changepoints",
    "changepoint_range",
    "yearly_seasonality",
    "weekly_seasonality",
    "daily_seasonality",
    "holidays",
    "seasonality_mode",
    "seasonality_prior_scale",
    "changepoint_prior_scale",
    "holidays_prior_scale",
    "mcmc_samples",
    "interval_width",
    "uncertainty_samples",
]

EXTRACT_IGNORE_PARAMS = ['changepoints']

SCORING_METRICS = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]


def get_prophet_init(**kwargs) -> Dict[str, any]:

    prophet_attrs = vars(Prophet())
    override_attrs = {
        attr: value
        for attr, value in prophet_attrs.items()
        if attr in BASE_PROPHET_ATTRS
    }
    override_attrs.update(
        (attr, value) for attr, value in kwargs.items() if attr in BASE_PROPHET_ATTRS
    )
    return override_attrs


def get_metrics(uncertainty_samples):
    metrics = deepcopy(SCORING_METRICS)
    if uncertainty_samples == 0:
        metrics.remove("coverage")
    return metrics


def get_extract_params():

    return [param for param in BASE_PROPHET_ATTRS if param not in EXTRACT_IGNORE_PARAMS]
