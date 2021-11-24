"""
This module contains private functions for handling configurations specific to a grouped wrapper
around the Prophet library.
"""


def _remove_coverage_metric_if_necessary(metrics, uncertainty_samples):
    """
    Private function for removing 'coverage' as a metric if the error estimates are never
    calculated (saves on runtime execution for cross validation)

    :param metrics: tuple of metrics to utilize in cross validation metric scoring
    :param uncertainty_samples: Model attribute (set at instantiation for Prophet) that specifies
           how to generate the error estimates: ['yhat_lower', 'yhat_upper']. If
           `uncertainty_samples` is set to 0, these fields will not be calculated, negating the
           ability for the `performance_metrics` function in Prophet from being able to
           calculate a 'coverage' metric.
    :return: the tuple of metrics to submit to Prophet's `performance_metrics` function
    """
    if uncertainty_samples == 0 and "coverage" in metrics:
        metrics = tuple(filter(lambda x: x != "coverage", metrics))
    return metrics
