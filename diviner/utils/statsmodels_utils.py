import pandas as pd
from diviner.config.constants import PREDICT_END_COL, PREDICT_START_COL


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
