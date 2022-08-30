import logging
import warnings
from collections import namedtuple
from typing import Tuple, Dict, List
from diviner.exceptions import DivinerException
import pandas as pd


def _restructure_fit_payload(train_results: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Restructuring function for converting a collection of dictionaries into a single dictionary
    for the final training grouped model structure.

    :param train_results: raw training (fit) results from a grouped model
    :return: dictionary of {master_grouping_key: model}
    """
    return {group_key: model for payload in train_results for group_key, model in payload.items()}


def _reorder_cols(df, key_columns: Tuple[str], master_grouping_key: str) -> pd.DataFrame:
    """
    Helper function for creating a user-friendly schema structure for the output prediction
    dataframe that mirrors what would be expected (grouping columns preceding data).

    :param df: dataframe with grouping columns appended to the predictions
    :param key_columns: the key column names that have been appended right-most to the prediction
                        dataframe.
    :param master_grouping_key: the master grouping key for setting the position of that column
    :return: a reformatted and modified schema of the prediction dataframe
    """
    masked_columns = [col for col in df.columns if col not in key_columns + [master_grouping_key]]
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


def _get_last_datetime_per_group(dt_indexed_group_data):
    return {group: df.index.max() for group, df in dt_indexed_group_data}


def _get_datetime_freq_per_group(dt_indexed_group_data):
    group_output = {}
    for group, df in dt_indexed_group_data:
        registered_freq = df.index.freq
        if registered_freq:
            group_output[group] = registered_freq
        else:
            group_output[group] = pd.infer_freq(df.index)
    return group_output


def _restrict_model_collection_by_groups(
    grouped_model,
    groups: List[Tuple[str]] = None,
):
    """
    Method for filtering group model(s) from a fit ``Diviner`` model, maintaining a key lookup
    failure state, and returning the filtered model collection and any failed lookup keys.
    The ``grouped_model`` object is a dictionary in the form of:
    {<grouping_key[tuple[str]]>: <fit instance model>} which is produced by a ``Diviner``
    framework wrapper implementation's ``fit`` method (i.e. ``GroupedProphet().fit()``.

    :param grouped_model: The instance attribute of a framework's implementation ``model``.
                          For example, in the use of ``GroupedProphet``, once ``.fit()`` is
                          called on the object, the object will have an attribute ``.model`` that
                          contains the dictionary of {<grouping_key>: model[Prophet]}. This
                          structure is what is passed to this argument.
    :param groups: The user-supplied list of tuples that defines the group entry values to
                   return predictions for. This is nullable (in the case of a user not submitting
                   a list of grouping keys, all models will return forecasts).
    :return: A named tuple of a subset of the matched groups and a list of unmatched keys, i.e.,:
             ({group[tuple[str]]: Prophet()}, [("a", "b"), ("e", "f"), ("z", "z")]) if the groups
             ("a", "b"), ("e", "f"), ("z", "z") were not present as grouping keys during the
             model fit execution.
    """

    Output = namedtuple("Output", "groups failures")
    key_lookup_failures = []
    if groups is not None:
        model_keys = set(grouped_model.keys())
        for group in groups:
            if group not in model_keys:
                key_lookup_failures.append(str(group))
        group_collection = {key: model for key, model in grouped_model.items() if key in groups}
    else:
        group_collection = grouped_model
    if len(group_collection) == 0:
        raise DivinerException(
            "Groups specified for subset forecasting are not present in the "
            "fit model. Validate model grouping keys by accessing "
            "<fit_model>.model.keys()"
        )
    return Output(group_collection, key_lookup_failures)


def _filter_groups_for_forecasting(
    grouped_model,
    groups: List[Tuple[str]] = None,
    on_error: str = "raise",
):
    """
    Method for error handling, logging, and passing filtering tasks to the private method
    ``_restrict_model_collection_by_groups()``. There are 3 modes of operation for error handling:
    * "raise" (default) will raise a  ``DivinerException`` if any user-supplied group does not
       exist in the fit model.
    * "warn" will log and print to stderr any unmatched grouping keys not present in the fit model.
    * "ignore" will silently ignore any unmatched keys and will only raise if none of the keys
      supplied match.

    :param grouped_model: The instance attribute of a framework's implementation ``model``.
                          For example, in the use of ``GroupedProphet``, once ``.fit()`` is
                          called on the object, the object will have an attribute ``.model`` that
                          contains the dictionary of {<grouping_key>: model[Prophet]}. This
                          structure is what is passed to this argument.
    :param groups: The user-supplied list of tuples that defines the group entry values to
                   return predictions for. This is nullable (in the case of a user not submitting
                   a list of grouping keys, all models will return forecasts).
    :param on_error: Alert level setting for handling mismatched group keys.
                     Default: ``"raise"``
                     The valid modes are:

                     * "ignore" - no logging or exception raising will occur if a submitted
                       group key in the ``groups`` argument is not present in the model object.

                       .. Note:: This is a silent failure mode and will not present any
                           indication of a failure to generate forecast predictions.

                     * "warn" - any keys that are not present in the fit model will be recorded
                       as logged warnings.
                     * "raise" - any keys that are not present in the fit model will cause
                       a ``DivinerException`` to be raised.
    :return: Filtered subset of grouped models to return for forecasting.
    """
    if groups is None:
        return grouped_model
    else:
        model_groups = _restrict_model_collection_by_groups(grouped_model, groups)

        if len(model_groups.failures) > 0:

            if on_error == "raise":
                raise DivinerException(
                    "Cannot perform predictions due to submitted group(s) "
                    f"missing from fit model: {','.join(model_groups.failures)}"
                )
            elif on_error == "warn":
                message = (
                    "Specified groups are unable to be predicted due to group(s) missing"
                    f"from fit model: {','.join(model_groups.failures)}"
                )
                warnings.warn(message, UserWarning)
                logger = logging.getLogger()
                logger.warning(message)

        return model_groups.groups
