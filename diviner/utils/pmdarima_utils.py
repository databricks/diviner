import warnings
import pandas as pd
from pmdarima.pipeline import Pipeline
from pmdarima.arima.auto import AutoARIMA
from pmdarima.arima.arima import ARIMA
from diviner.exceptions import DivinerException

_PMDARIMA_MODEL_PARAMS = {
    "order",
    "seasonal_order",
    "start_params",
    "method",
    "maxiter",
    "out_of_sample_size",
    "scoring",
    "trend",
    "with_intercept",
    "sarimax_kwargs",
}
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
            f"Failed to access the model instance type. '{type(model)}' "
            "is not recognized."
        )
    return arima_model  # Access the wrapped ARIMA model instance


def _get_arima_params(arima_model):

    params = {}
    for param in _PMDARIMA_MODEL_PARAMS:
        try:
            value = getattr(arima_model, param)
            if param in _COMPOUND_KEYS.keys():
                extracted = dict(zip(_COMPOUND_KEYS[param], value))
                params.update(extracted)
            else:
                params[param] = value
        except AttributeError as e:
            warnings.warn(f"Cannot extract parameter '{param}' from model. {e}")

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


def generate_prediction_config(
    group_keys,
    group_key_columns,
    n_periods,
    alpha=0.05,
    return_conf_int=False,
    inverse_transform=True,
):
    """

    :param group_keys:
    :param group_key_columns:
    :param n_periods:
    :param alpha:
    :param return_conf_int:
    :param inverse_transform:
    :return:
    """
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
