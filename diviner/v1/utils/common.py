from typing import Tuple, Dict, List
from diviner.v1.exceptions import DivinerException
import pandas as pd
from collections import namedtuple
from diviner.v1.config.constants import PREDICT_END_COL, PREDICT_START_COL


def _restructure_fit_payload(train_results: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Restructuring function for converting a collection of dictionaries into a single dictionary
    for the final training grouped model structure.

    :param train_results: raw training (fit) results from a grouped model
    :return: dictionary of {master_grouping_key: model}
    """
    return {
        group_key: model
        for payload in train_results
        for group_key, model in payload.items()
    }


def _reorder_cols(
    df, key_columns: Tuple[str], master_grouping_key: tuple
) -> pd.DataFrame:
    """
    Helper function for creating a user-friendly schema structure for the output prediction
    dataframe that mirrors what would be expected (grouping columns preceding data).

    :param df: dataframe with grouping columns appended to the predictions
    :param key_columns: the key column names that have been appended right-most to the prediction
                        dataframe.
    :param master_grouping_key: the master grouping key for setting the position of that column
    :return: a reformatted and modified schema of the prediction dataframe
    """
    masked_columns = [
        col for col in df.columns if col not in key_columns + [master_grouping_key]
    ]
    reordered_df = df[[master_grouping_key] + key_columns + masked_columns]
    return reordered_df


def _restructure_predictions(prediction_dfs, key_columns, master_key):
    """
    Restructuring function to create the final prediction output dataframe from a collection
    of dataframes, adding in filtering fields to the prediction to persist the original
    structure of the submitted data.

    :param prediction_dfs: The per-group dataframes of forecasted predictions
    :param key_columns: Names of the grouping key columns that were defined during training
    :param master_key: The master grouping key column name
    :return: A dataframe of all predictions with grouping key columns and index values
             added as reference columns
    """

    prediction_output = pd.concat(prediction_dfs).reset_index(drop=True)
    prediction_output[key_columns] = pd.DataFrame(
        prediction_output[master_key].tolist(), index=prediction_output.index
    )
    prediction_output.insert(
        0,
        f"{master_key}_columns",
        prediction_output.apply(lambda x: tuple(key_columns), axis=1),
    )
    reordered = _reorder_cols(prediction_output, key_columns, f"{master_key}_columns")
    cleaned = reordered.copy()
    cleaned.drop(columns=[master_key], axis=1, inplace=True)
    return cleaned


def _validate_keys_in_df(df, key_columns: Tuple):
    """
    Validation function for ensuring that the grouping keys that are passed in to the
    class constructor are present in the passed-in DataFrame.

    :param df: DataFrame of grouping keys and data for training or inference
    :param key_columns: args-based key columns passed in for grouping on
    :return: None
    """

    columns_list = list(df.columns)
    if not set(key_columns).issubset(set(columns_list)):
        raise DivinerException(
            f"Not all key grouping columns supplied: {key_columns} are present "
            f"in the submitted df: {columns_list}"
        )


def validate_prediction_config_df(df, key_columns: Tuple):

    _validate_keys_in_df(df, key_columns)
    columns_set = set(df.columns)
    validation = {PREDICT_START_COL, PREDICT_END_COL}
    if not validation.issubset(columns_set):
        raise DivinerException(
            f"Prediction configuration DataFrame columns: `{columns_set}` "
            f"do not contain the required columns: `{validation}`"
        )


def create_reporting_df(extract_dict, master_key, group_key_columns):
    """
    Structural consolidation extract for a grouped model to generate an MLflow
    artifact-compatible representation of each of the group's model attributes (metrics or
    params) for a single run.
    :param extract_dict: Extracted attributes from a model
    :param master_key: The master grouping key column name
    :param group_key_columns: The names of the grouping key columns used to train
                              a single instance of a grouped model
    :return: A Pandas DataFrame containing the attributes, grouping keys, and master grouping key
             as columns with a row for each unique group's model.
    """
    base_df = pd.DataFrame.from_dict(extract_dict).T.sort_index(inplace=False)
    base_df[master_key] = base_df.index.to_numpy()
    base_df.index.names = group_key_columns
    extracted_df = base_df.reset_index(inplace=False)
    extracted_df.insert(
        0,
        f"{master_key}_columns",
        extracted_df.apply(lambda x: tuple(group_key_columns), axis=1),
    )
    extracted_df.drop(columns=[master_key], axis=1, inplace=True)
    return extracted_df


def generate_forecast_horizon_series(
    max_datetime_per_group, horizon: int, frequency: str = None
):
    """
    Utility function for generating a forecasting series based on user-supplied attributes.
    This function will begin the forecasting series based on the next frequency-based event from
    the training data that was used for model creation.
    For example:
    A training data set consists of {"2021-01-01", "2021-01-02", ... "2021-04-01"} as its
    datetime values for a group {(a, 2)}.
    The argument `max_datetime_per_group` will, for the entry for the grouping key {(a, 2)}
    have a value of "2021-04-01".
    This function will increment to the next `frequency` value following this datetime entry.
    If, for example, the frequency is set to "D", the start of the forecast horizon time series
    for this group {(a, 2)} will be "2021-04-02" and will continue at the "D" (day) frequency
    for as many events as are specified by the `horizon` argument.
    :param max_datetime_per_group: Dictionary of {master_group_key: max date of training series}
    :param horizon: The number of discrete datetime events to generate for the forecast.
    :param frequency: The boundary between events being generated (as well as the boundary to
                      the max datetime value from each group's training data that defines the
                      start of the forecast series). These values must be of Pandas date_range
                      format.
    :return: A named tuple `TimeBoundaries` that contains the entries (start, end) that define
             the starting datetime value and the ending datetime value for a given model's
             prediction threshold.
    """
    TimeBoundaries = namedtuple("TimeBoundaries", "start end")
    forecast_series = {}
    for group, max_date in max_datetime_per_group.items():
        if not frequency:
            frequency = max_date.freqstr
        start_datetime = pd.date_range(start=max_date, periods=2, freq=frequency)[-1]
        generated_series = pd.date_range(
            start=start_datetime, periods=horizon, freq=frequency
        )
        forecast_series[group] = TimeBoundaries(
            generated_series[0], generated_series[-1]
        )
    return forecast_series


def convert_forecast_horizon_series_to_df(forecast_series, group_key_columns):
    """

    :param forecast_series:
    :param group_key_columns:
    :return:
    """
    collection = []
    for key, value in forecast_series.items():
        row = {}
        for index, column in enumerate(group_key_columns):
            row[column] = key[index]
        row["start"] = value.start
        row["end"] = value.end
        collection.append(row)
    return pd.DataFrame.from_records(collection)
