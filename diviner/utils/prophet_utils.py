from inspect import signature
from typing import Tuple, List

from prophet.diagnostics import cross_validation, performance_metrics

from diviner.config.grouped_prophet.prophet_config import _get_extract_params
from diviner.config.grouped_prophet.utils import prophet_config_utils
from diviner.utils.common import _filter_groups_for_forecasting


def _generate_future_df(model, horizon, frequency):
    """
    Internal helper wrapper around the Prophet internal make_future_dataframe class method
    that will generate a series of rows of future datatime entries based on where the
    training dataset time horizon left off.

    :param model: A fit Prophet model
    :param horizon: integer of units of datetime period to generate
    :param frequency: the datetime period type to generate in Pandas `date_range` offset
                        alias format
    :return: A dataframe containing a list of datetime entries that starts after the last
             remaining datetime event from the training data, continues at a span defined by
             `period_type` for `horizon` number of iterations.
    """
    return model.make_future_dataframe(periods=horizon, freq=frequency, include_history=False)


def generate_future_dfs(
    grouped_model,
    horizon: int,
    frequency: str,
    groups: List[Tuple[str]] = None,
    on_error: str = "raise",
):
    """
    Utility function for continuing from where the training dataframe left off and generating
    `horizon` number of `period_type` row entries per group that were in the training data set.

    :param grouped_model: A fit grouped Prophet model
    :param horizon: integer of units of datetime period to generate
    :param frequency: the datetime period type to generate in Pandas `date_range` offset
                        alias format
    see: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :param groups: ``List[Tuple[str]]`` the collection of
                   group(s) to generate forecast predictions. The group definitions must be
                   the values within the ``group_key_columns`` that were used during the
                   ``fit`` of the model in order to return valid forecasts.

                   .. Note:: The positional ordering of the values are important and must match
                     the order of ``group_key_columns`` for the ``fit`` argument to provide
                     correct prediction forecasts.
    :param on_error: Alert level setting for handling mismatched group keys if groups is set.
                     Default: ``"raise"``
                     Must be one of "ignore", "warn", or "raise"
    :return: A collection of tuples of master grouping key and future dataframes
    """

    model_collection = _filter_groups_for_forecasting(grouped_model, groups, on_error)

    df_collection = {
        key: _generate_future_df(model, horizon, frequency)
        for key, model in model_collection.items()
    }

    return list(df_collection.items())


def _cross_validate_and_score_model(
    model,
    horizon,
    period=None,
    initial=None,
    parallel=None,
    cutoffs=None,
    metrics=None,
    **kwargs,
):
    """
    Wrapper around Prophet's `cross_validation` and `performance_metrics` functions within
    the `prophet.diagnostics` module.
    Provides backtesting metric evaluation based on the configurations specified for
    initial, horizon, and period (optionally, a specified 'cutoffs' list of DateTime or string
    date-time entries can override the backtesting split boundaries for training and validation).

    :param model: Prophet model instance that has been fit
    :param horizon: String pd.Timedelta format that defines the length of forecasting values
                    to generate in order to acquire error metrics.
                    examples: '30 days', '1 year'
     :param period: the periodicity of how often a windowed validation will occur. Default is
                    0.5 * horizon value.
    :param initial: The minimum amount of training data to include in the first cross validation
                    window.
    :param parallel: mode of computing cross validation statistics. (None, processes, or threads)
    :param cutoffs: List of pd.Timestamp values that specify cutoff overrides to be used in
                    conducting cross validation.
    :param metrics: List of metrics to evaluate and return for the provided model
                    note: see supported metrics in Prophet documentation:
                    https://facebook.github.io/prophet/docs/diagnostics.html#cross-validation
    :param kwargs: cross validation overrides for Prophet's implementation of metric evaluation.
    :return: Dict[str, float] of each metric and its value averaged over each time horizon.
    """

    if metrics:
        metrics = prophet_config_utils._remove_coverage_metric_if_necessary(
            metrics, model.uncertainty_samples
        )

    # extract `performance_metrics` *args if present
    performance_metrics_defaults = signature(performance_metrics).parameters

    performance_metrics_args = {}
    for param, value in performance_metrics_defaults.items():
        if value.default != value.empty and value.name != "metrics":
            performance_metrics_args[param] = kwargs.pop(param, value.default)

    model_cv = cross_validation(
        model=model,
        horizon=horizon,
        period=period,
        initial=initial,
        parallel=parallel,
        cutoffs=cutoffs,
        disable_tqdm=kwargs.pop("disable_tqdm", True),
    )
    horizon_metrics = performance_metrics(model_cv, metrics=metrics, **performance_metrics_args)

    return {
        metric: horizon_metrics[metric].mean()
        for metric in list(horizon_metrics.columns)
        if metric != "horizon"
    }


def _extract_params(model):
    """
    Helper function for retrieving the model parameters from a single Prophet model.

    :param model: A trained (fit) Prophet model instance
    :return: Dict[str, any] of tunable parameters for a Prophet model
    """

    return {param: getattr(model, param) for param in _get_extract_params()}
