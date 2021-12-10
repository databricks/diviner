import warnings

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


def _extract_arima_model_from_pipeline(pipeline):
    arima_model = pipeline.steps[-1][
        1
    ]  # The AutoARIMA model object must be the last element.
    return arima_model.model_  # Access the wrapped ARIMA model instance


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
