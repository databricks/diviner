import warnings
from typing import Tuple, List
from copy import deepcopy

import pandas as pd
from pmdarima import AutoARIMA, ARIMA
from pmdarima.pipeline import Pipeline

from diviner.utils.common import _filter_groups_for_forecasting
from diviner.exceptions import DivinerException

_COMPOUND_KEYS = {"order": ("p", "d", "q"), "seasonal_order": ("P", "D", "Q", "s")}
_PMDARIMA_MODEL_METRICS = {"aic", "aicc", "bic", "hqic", "oob"}


def _extract_arima_model(model):
    if isinstance(model, Pipeline):
        pipeline_model = model.steps[-1][1]
        if isinstance(pipeline_model, AutoARIMA):
            arima_model = pipeline_model.model_
        elif isinstance(pipeline_model, ARIMA):
            arima_model = pipeline_model
        else:
            raise DivinerException(
                f"The final stage of the submitted Pipeline: '{type(pipeline_model).__name__}' "
                "is not able to be accessed. Must be either 'AutoARIMA' or 'ARIMA'."
            )
    elif isinstance(model, AutoARIMA):
        arima_model = model.model_
    elif isinstance(model, ARIMA):
        arima_model = model
    else:
        raise DivinerException(
            f"Failed to access the model instance type. '{type(model)}' " "is not recognized."
        )
    return arima_model  # Access the wrapped ARIMA model instance


def _get_arima_params(arima_model):

    params = {
        key: value
        for key, value in arima_model.get_params().items()
        if key not in _COMPOUND_KEYS.keys()
    }
    for key, value in _COMPOUND_KEYS.items():
        compound_param = getattr(arima_model, key)
        params.update(dict(zip(value, compound_param)))

    return params


def _get_arima_training_metrics(arima_model):

    metrics = {}
    for metric in _PMDARIMA_MODEL_METRICS:
        try:
            value = getattr(arima_model, metric)()  # these are functions
            if value:
                metrics[metric] = value
        except AttributeError as e:
            warnings.warn(f"Cannot extract metric '{metric}' from model. {e}")
    return metrics


def _construct_prediction_config(
    group_keys,
    group_key_columns,
    n_periods,
    alpha=0.05,
    return_conf_int=False,
    inverse_transform=False,
):
    config = []
    for key in group_keys:
        row = {
            "n_periods": n_periods,
            "alpha": alpha,
            "return_conf_int": return_conf_int,
            "inverse_transform": inverse_transform,
        }
        for idx, col in enumerate(group_key_columns):
            row[col] = key[idx]
        config.append(row)
    return pd.DataFrame.from_records(config)


def _generate_prediction_config(
    grouped_pmdarima_model,
    n_periods,
    alpha=0.05,
    return_conf_int=False,
    inverse_transform=False,
):

    model_group_keys = list(grouped_pmdarima_model.model.keys())
    group_key_columns = grouped_pmdarima_model._group_key_columns
    return _construct_prediction_config(
        model_group_keys,
        group_key_columns,
        n_periods,
        alpha,
        return_conf_int,
        inverse_transform,
    )


def _generate_prediction_datetime_series(fit_max_datetime, fit_freq, periods):

    series_start = pd.date_range(start=fit_max_datetime, periods=2, freq=fit_freq)[-1]
    return pd.date_range(start=series_start, periods=periods, freq=fit_freq)


def _generate_group_subset_prediction_config(
    grouped_pmdarima_model,
    groups: List[Tuple[str]],
    n_periods: int,
    alpha: float = 0.05,
    return_conf_int: bool = False,
    inverse_transform: bool = False,
    on_error: str = "raise",
):
    model_copy = deepcopy(grouped_pmdarima_model)
    model_copy.model = _filter_groups_for_forecasting(
        grouped_pmdarima_model.model, groups, on_error
    )

    return _generate_prediction_config(
        model_copy, n_periods, alpha, return_conf_int, inverse_transform
    )
