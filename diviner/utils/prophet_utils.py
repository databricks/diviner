from diviner.config.grouped_prophet.prophet_config import (
    get_base_metrics,
    get_extract_params,
_validate_user_metrics
)
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import pandas as pd
from inspect import signature


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
    return model.make_future_dataframe(
        periods=horizon, freq=frequency, include_history=False
    )


def generate_future_dfs(grouped_model, horizon: int, frequency: str):
    """
    Utility function for continuing from where the training dataframe left off and generating
    `horizon` number of `period_type` row entries per group that were in the training data set.
    :param grouped_model: A fit grouped Prophet model
    :param horizon: integer of units of datetime period to generate
    :param frequency: the datetime period type to generate in Pandas `date_range` offset
                        alias format
    see: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :return: A collection of tuples of master grouping key and future dataframes
    """
    df_collection = {
        key: _generate_future_df(model, horizon, frequency)
        for key, model in grouped_model.items()
    }

    return list(df_collection.items())


def cross_validate_model(model, metrics=None, **kwargs):
    """
    Wrapper around Prophet's `cross_validation` and `performance_metrics` functions within
    the `prophet.diagnostics` module.
    Provides backtesting metric evaluation based on the configurations specified for
    initial, horizon, and period (optionally manual 'cutoffs' as well).
    :param model: Prophet model instance that has been fit
    :param metrics: List of metrics to evaluate and return for the provided model
                    note: Metrics not a member of:
                    `["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]`
                    will raise a DivinerException.
    :param kwargs: cross validation overrides for Prophet's implementation of backtesting.
                   note: two of the potential kwargs entries that are contained here can be for the
                   *args defaulted values within Prophet's `performance_metrics` function:
                   'rolling_window' and 'monthly' which specify how to roll up the
                   'raw' data returned from the `cross_validation` function. If not specified
                   within kwargs, they will retain their defaulted values from within Prophet.
    :return: Dict[str, float] of each metric and its averaged value over each time horizon.
    """
    # Instead of populating 'NaN' for invalid metrics, raise an Exception.
    if metrics:
        cv_metrics = _validate_user_metrics(metrics, model.uncertainty_samples)
    else:
        cv_metrics = get_base_metrics(uncertainty_samples=model.uncertainty_samples)

    # extract `performance_metrics` *args if present
    performance_metrics_defaults = signature(performance_metrics).parameters
    rolling_window = kwargs.pop(
        "rolling_window", performance_metrics_defaults["rolling_window"].default
    )
    monthly = kwargs.pop("monthly", performance_metrics_defaults["monthly"].default)

    model_cv = cross_validation(
        model=model, disable_tqdm=kwargs.pop("disable_tqdm", True), **kwargs
    )
    horizon_metrics = performance_metrics(
        model_cv, metrics=cv_metrics, rolling_window=rolling_window, monthly=monthly
    )

    return {
        metric: horizon_metrics[metric].mean() if metric in horizon_metrics else np.nan
        for metric in cv_metrics
    }


def extract_params(model):
    """
    Helper function for retrieving the model parameters from a single Prophet model.
    :param model: A trained (fit) Prophet model instance
    :return: Dict[str, any] of tunable parameters for a Prophet model
    """

    return {param: getattr(model, param) for param in get_extract_params()}


def create_reporting_df(extract_dict, master_key, group_key_columns):
    """
    Structural consolidation extract for the GroupedProphet model to generate an MLflow
    artifact-compatible representation of each of the group's model attributes (metrics or
    params) for a single run.
    :param extract_dict: Extracted attributes from a Prophet model
    :param master_key: The master grouping key column name
    :param group_key_columns: The names of the grouping key columns used to train
                              and instance of GroupedProphet model
    :return: A Pandas DataFrame containing the attributes, grouping keys, and master grouping key
             as columns with a row for each unique group's model.
    """
    base_df = pd.DataFrame.from_dict(extract_dict).T.sort_index(inplace=False)
    base_df[master_key] = base_df.index.to_numpy()
    base_df.index.names = group_key_columns
    extracted_df = base_df.reset_index(inplace=False)
    return extracted_df
