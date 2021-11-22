import logging
import warnings
from diviner.exceptions import DivinerException

_STATSMODELS_METRICS = {"aic", "aicc", "bic", "hqic", "mae", "mse", "sse"}


def _validate_metrics(metrics):

    if metrics:
        if set(metrics).issubset(_STATSMODELS_METRICS):
            return metrics
        else:
            msg = (
                f"Metric(s) supplied are invalid: {metrics}. "
                f"Must be member(s) of: {_STATSMODELS_METRICS}"
            )
            raise DivinerException(msg)
    else:
        return _STATSMODELS_METRICS


def _extract_statsmodels_single_model_metrics(model, metrics, warning):

    if not warning:
        logging.captureWarnings(~warning)

    scores = {}
    for metric in metrics:
        try:
            metric_value = getattr(model, metric)
            if isinstance(metric_value, float):
                scores[metric] = metric_value
        except AttributeError:
            warnings.warn(
                f"Metric '{metric}' could not be retrieved from the model",
                RuntimeWarning,
                stacklevel=4,  # don't show the user protected methods in warning log
            )
    if not scores:
        warnings.warn(
            f"No metrics were able to be extracted from the model. "
            f"Attempted metrics to extract: {metrics}",
            RuntimeWarning,
            stacklevel=4,  # don't show the user protected methods in warning log
        )
    return scores


def _extract_statsmodels_metrics(grouped_model, metrics, warning):

    metrics = _validate_metrics(metrics)
    extract = {
        group_key: _extract_statsmodels_single_model_metrics(model, metrics, warning)
        for group_key, model in grouped_model.items()
    }

    return extract
