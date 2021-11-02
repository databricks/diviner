import importlib
import inspect
from diviner.exceptions import DivinerException


def get_statsmodels_model(model_name: str):

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
