from prophet import Prophet
from copy import deepcopy
from inspect import signature
from diviner.exceptions import DivinerException

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


def _validate_user_metrics(metrics, uncertainty_samples):

    if not set(metrics).issubset(set(SCORING_METRICS)):
        raise DivinerException(
            f"Metrics supplied for cross validation: {metrics} contains"
            f"invalid entries. Metrics must be part of: {SCORING_METRICS}"
        )
    else:
        if uncertainty_samples == 0 and "coverage" in metrics:
            metrics.remove("coverage")
        return metrics

