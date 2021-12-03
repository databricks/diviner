from prophet.diagnostics import cross_validation, performance_metrics
from diviner.config.grouped_prophet.utils import prophet_config_utils


def group_cross_validation(
    grouped_model, horizon, period=None, initial=None, parallel=None, cutoffs=None
):
    """
    Model debugging utility function for performing cross validation for each model within the
    GroupedProphet collection.
    note: the output of this will be a Pandas DataFrame for each grouping key per cutoff
    boundary in the datetime series. The output of this function will be many times larger than
    the original input data utilized for training of the model.

    :param grouped_model: A fit GroupedProphet model
    :param horizon: pd.Timedelta formatted string (i.e. "14 days" or "18 hours") to define the
                    amount of time to utilize for a validation set to be created.
    :param period: the periodicity of how often a windowed validation will occur. Default is
                    0.5 * horizon value.
    :param initial: The minimum amount of training data to include in the first cross validation
                    window.
    :param parallel: mode of computing cross validation statistics. (None, processes, or threads)
    :param cutoffs: List of pd.Timestamp values that specify cutoff overrides to be used in
                    conducting cross validation.
    :return: Dictionary of {group_key: cross validation Pandas DataFrame}
    """
    return {
        group_key: cross_validation(
            model, horizon, period, initial, parallel, cutoffs, disable_tqdm=True
        )
        for group_key, model in grouped_model.model.items()
    }


def _single_model_performance_evaluation(cv_df, metrics, rolling_window, monthly):

    return performance_metrics(cv_df, metrics, rolling_window, monthly)


def group_performance_metrics(
    cv_results, grouped_model, metrics=None, rolling_window=0.1, monthly=False
):
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
    :param rolling_window: Defines how much data to use in each rolling window as a range of [0, 1]
                            for computing the performance metrics.
    :param monthly: If set to true, will collate the windows to ensure that horizons are computed
                    as number of months from the cutoff date. Only useful for date data that has
                    yearly seasonality associated with calendar day of month.
    :return: Dictionary of {group_key: performance metrics per window Pandas DataFrame}
    """

    grouped_metrics = {}

    for group_key, model in grouped_model.model.items():

        cv_df = cv_results[group_key]
        if metrics:
            metrics = prophet_config_utils._remove_coverage_metric_if_necessary(
                metrics, model.uncertainty_samples
            )

        grouped_metrics[group_key] = _single_model_performance_evaluation(
            cv_df, metrics, rolling_window, monthly
        )

    return grouped_metrics
