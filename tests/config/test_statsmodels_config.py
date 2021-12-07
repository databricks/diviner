import pytest
import inspect
from importlib import import_module
from diviner.config.grouped_statsmodels.statsmodels_config import (
    _get_statsmodels_model,
    _extract_fit_kwargs,
)
from diviner.exceptions import DivinerException


@pytest.mark.parametrize(
    "model",
    (
        "ARIMA",
        "arima",
        "SARIMAX",
        "sarimax",
        "Holt",
        "DynamicFactor",
        "UECM",
        "VAR",
        "ExponentialSmoothing",
    ),
)
def test_valid_models_returned(model):

    ATTRS = {"fit", "predict"}
    clazz = _get_statsmodels_model(model)
    members = [key for key, value in inspect.getmembers(clazz, inspect.isfunction)]
    for attr in ATTRS:
        assert attr in members


@pytest.mark.parametrize("invalid", ("prophet", "NotRealModel"))
def test_invalid_model_type_raises(invalid):
    available_classes = {
        key: ref
        for key, ref in inspect.getmembers(
            import_module("statsmodels.tsa.api"), inspect.isclass
        )
    }
    with pytest.raises(
        DivinerException,
        match=f"Statsmodels model type '{invalid}' is not a valid model type. "
        f"Must be one of: {', '.join(list(available_classes.keys()))}",
    ):
        _get_statsmodels_model(invalid)


def test_extract_fit_kwargs():

    model_class = _get_statsmodels_model("arima")
    fit_kwargs = {"transformed": False, "return_params": True, "low_memory": True}
    clazz_kwargs = {"enforce_stationarity": False, "concentrate_scale": True}

    extracted_kwargs = _extract_fit_kwargs(
        model_class,
        transformed=False,
        return_params=True,
        low_memory=True,
        enforce_stationarity=False,
        concentrate_scale=True,
    )

    assert extracted_kwargs.clazz == clazz_kwargs
    assert extracted_kwargs.fit == fit_kwargs
