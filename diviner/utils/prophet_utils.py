from diviner.config.grouped_prophet.prophet_config import (
    get_metrics,
    get_extract_params,
)
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import pandas as pd


def _generate_future_df(model, horizon, period_type):
    """
    Internal helper wrapper around the Prophet internal make_future_dataframe class method
    that will generate a series of rows of future datatime entries based on where the
    training dataset time horizon left off.
    :param model: A fit Prophet model
    :param horizon: integer of units of datetime period to generate
    :param period_type: the datetime period type to generate in Pandas `date_range` offset
                        alias format
    :return: A dataframe containing a list of datetime entries that starts after the last
             remaining datetime event from the training data, continues at a span defined by
             `period_type` for `horizon` number of iterations.
    """
    return model.make_future_dataframe(
        periods=horizon, freq=period_type, include_history=False
    )


def generate_future_dfs(grouped_model, horizon: int, period_type: str):
    """
    Utility function for continuing from where the training dataframe left off and generating
    `horizon` number of `period_type` row entries per group that were in the training data set.
    :param grouped_model: A fit grouped Prophet model
    :param horizon: integer of units of datetime period to generate
    :param period_type: the datetime period type to generate in Pandas `date_range` offset
                        alias format
    see: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :return: A collection of tuples of master grouping key and future dataframes
    """
    df_collection = {
        key: _generate_future_df(model, horizon, period_type)
        for key, model in grouped_model.items()
    }

    return list(df_collection.items())


def cross_validate_model(model, cv_initial, cv_period, cv_horizon, parallel):
    metrics = get_metrics(uncertainty_samples=model.uncertainty_samples)
    model_cv = cross_validation(
        model=model,
        initial=cv_initial,
        period=cv_period,
        horizon=cv_horizon,
        parallel=parallel,
    )
    horizon_metrics = performance_metrics(model_cv, metrics=metrics)

    return {
        metric: horizon_metrics[metric].mean() if metric in horizon_metrics else np.nan
        for metric in metrics
    }


def extract_params(model):

    return {param: getattr(model, param) for param in get_extract_params()}


def create_reporting_df(extract_dict, master_key, group_key_columns):
    base_df = pd.DataFrame.from_dict(extract_dict).T.sort_index(inplace=False)
    base_df[master_key] = base_df.index.to_numpy()
    base_df.index.names = group_key_columns
    extracted_df = base_df.reset_index(inplace=False)
    return extracted_df
