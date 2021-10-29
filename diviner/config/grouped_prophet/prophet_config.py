from prophet import Prophet
from copy import deepcopy
from inspect import signature

SCORING_METRICS = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]


def get_base_metrics(uncertainty_samples):
    metrics = deepcopy(SCORING_METRICS)
    if uncertainty_samples == 0:
        metrics.remove("coverage")
    return metrics


def get_extract_params():

    prophet_signature = list(signature(Prophet).parameters.keys())
    prophet_signature.remove("changepoints")

    return prophet_signature
