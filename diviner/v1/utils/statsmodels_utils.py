import pandas as pd
import numpy as np
from diviner.v1.config.constants import PREDICT_END_COL, PREDICT_START_COL


def _get_max_datetime_per_group(dt_indexed_group_data):

    continuation_datetimes = {
        group: df.index.max() for group, df in dt_indexed_group_data
    }

    return continuation_datetimes


def _resolve_forecast_duration_var_model(group_entry):

    group_entry[PREDICT_START_COL] = pd.to_datetime(group_entry[PREDICT_START_COL])
    group_entry[PREDICT_END_COL] = pd.to_datetime(group_entry[PREDICT_END_COL])
    group_entry["delta"] = pd.to_timedelta(
        group_entry[PREDICT_END_COL] - group_entry[PREDICT_START_COL]
    )
    delta = group_entry["delta"]
    resolution = delta.resolution_string
    if resolution == "D":
        return delta.days
    elif resolution == "H":
        return delta.hours
    elif resolution == "M":
        return delta.minutes
    else:
        return delta.seconds


def _extract_params_from_model(model):

    model_params_attr = getattr(model._results, "params")

    # Cast np.ndarray to list
    list_convert = {
        key: value if not isinstance(value, np.ndarray) else list(value)
        for key, value in model_params_attr.items()
    }
    # Cast nan to None
    param_extract = {
        key: value if value == value else None for key, value in list_convert.items()
    }
    return param_extract
