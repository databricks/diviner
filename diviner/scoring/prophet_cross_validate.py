from prophet.diagnostics import cross_validation, performance_metrics
from diviner.config.grouped_prophet.utils import prophet_config_utils


def group_cross_validation(grouped_model, **kwargs):
    """
    Model debugging utility function for performing cross validation for each model within the
    GroupedProphet collection.
    note: the output of this will be a Pandas DataFrame for each grouping key per cutoff
    boundary in the datetime series. The output of this function will be many times larger than
    the original input data utilized for training of the model.

    :param grouped_model: A fit GroupedProphet model
    :param kwargs: Cross validation overrides for the signature of Prophet's
                   `prophet.diagnostics.cross_validation`: (e.g., 'horizon', 'period',
                   'initial', etc.)
    :return: Dictionary of {group_key: cross validation Pandas DataFrame}
    """
    return {
        group_key: cross_validation(model, **kwargs)
        for group_key, model in grouped_model.model.items()
    }


def _single_model_performance_evaluation(cv_df, metrics, **kwargs):

    return performance_metrics(cv_df, metrics, **kwargs)


def group_performance_metrics(cv_results, grouped_model, metrics=None, **kwargs):
    """
    Model debugging utility function for evaluating performance metrics from the grouped
    cross validation extract.
    This will output a metric table for each time boundary from cross validation, for each model.
    note: This output will be very large and is intended to be used as a debugging tool only.

    :param cv_results: The output return of `group_cross_validation`
    :param grouped_model: The fit model used in `group_cross_validation` to generate the cv_results
                          data collection
    :param metrics: (Optional) overrides (subtractive) for metrics to generate for this function's
                    output.
                    note: see supported metrics in Prophet documentation:
                    https://facebook.github.io/prophet/docs/diagnostics.html#cross-validation
                    note: any model in the collection that was fit with the argument
                    `uncertainty_samples` set to '0' will have the metric 'coverage' removed from
                    evaluation due to the fact that `yhat_error` values are not calculated with
                    that configuration of that parameter.
    :param kwargs: overrides to Prophet's `performance_metrics` function (`rolling_window`
                   and `monthly`)
    :return: Dictionary of {group_key: performance metrics per window Pandas DataFrame}
    """

    grouped_metrics = {}

    for group_key, model in grouped_model.model.items():

        cv_df = cv_results[group_key]
        if metrics:
            metrics = prophet_config_utils._reconcile_metrics(
                metrics, model.uncertainty_samples
            )

        grouped_metrics[group_key] = _single_model_performance_evaluation(
            cv_df, metrics, **kwargs
        )

    return grouped_metrics
