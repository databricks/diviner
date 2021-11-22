import importlib
import inspect
from diviner.exceptions import DivinerException
from collections import namedtuple


def _get_statsmodels_model(model_name: str):

    model_name_lowered = str.lower(model_name)
    statsmodels_api = importlib.import_module("statsmodels.tsa.api")
    api_members = inspect.getmembers(statsmodels_api, inspect.isclass)
    clazzes = {str.lower(clazz): clazz for clazz, ref in api_members}
    if model_name_lowered not in list(clazzes.keys()):
        raise DivinerException(
            f"Statsmodels model type '{model_name}' is not a valid model type. "
            f"Must be one of: {clazzes}"
        )

    clazz = getattr(statsmodels_api, clazzes[model_name_lowered])

    return clazz


def _extract_fit_kwargs(clazz, **kwargs):
    """
    Helper function for separating out class and fit method kwargs for a particular model
    :param clazz: The class instance of the user-chosen model
    :param kwargs: kwargs passed in to the grouped model instance for class and fit overrides
    :return:
    """
    kwarg_payload = namedtuple("KwargPayload", "clazz fit")
    fit_kwargs = {}
    clazz_signature = inspect.signature(clazz).parameters.keys()
    for key, value in kwargs.items():
        if (
            key in inspect.signature(clazz.fit).parameters.keys()
            and key not in clazz_signature
        ):
            fit_kwargs[key] = value
    if fit_kwargs:
        for key in fit_kwargs.keys():
            kwargs.pop(key)

    return kwarg_payload(kwargs, fit_kwargs)
