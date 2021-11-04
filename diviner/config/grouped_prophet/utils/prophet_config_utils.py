from diviner.exceptions import DivinerException

_SCORING_METRICS = ("mse", "rmse", "mae", "mape", "mdape", "smape", "coverage")


def _reconcile_metrics(metrics, uncertainty_samples):
    """
    Private function for removing 'coverage' as a metric if the error estimates are never
    calculated (saves on runtime execution for cross validation)
    :param metrics: tuple of metrics to utilize in cross validation metric scoring
    :param uncertainty_samples: Model attribute (set at instantiation for Prophet) that specifies
           how to generate the error estimates: ['yhat_lower', 'yhat_upper']. If
           `uncertainty_samples` is set to 0, these fields will not be calculated, negating the
           ability for the `performance_metrics` function in Prophet from being able to
           calculate a 'coverage' metric.
    :return:
    """
    if uncertainty_samples == 0 and "coverage" in metrics:
        metrics = tuple(filter(lambda x: x != "coverage", metrics))
    return metrics


def _validate_user_metrics(metrics):
    """
    Private function to validate user-supplied metrics are contained within `_SCORING_METRICS` to
    prevent `NaN` values from being recorded in the final metrics summary extract.
    :param metrics: iterable of metric names supplied by user configuration from
                    the signature argument of `GroupedProphet().cross_validation()`
    :return: Validated subset of allowed metrics
    """
    if not set(metrics).issubset(set(_SCORING_METRICS)):
        raise DivinerException(
            f"Metrics supplied for cross validation: {metrics} contains"
            f"invalid entries. Metrics must be part of: {_SCORING_METRICS}"
        )
    else:
        return metrics
