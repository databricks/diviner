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
    return {
        group_key: model
        for payload in train_results
        for group_key, model in payload.items()
    }


def _reorder_cols(
    df, key_columns: Tuple[str], master_grouping_key: str
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
        prediction_output.apply(lambda x: key_columns, axis=1),
    )
    reordered = _reorder_cols(prediction_output, key_columns, f"{master_key}_columns")
    reordered.drop(columns=[master_key], axis=1, inplace=True)
    return reordered


def _fit_check(fn):
    """
    Model fit validation decorator. Performs a check to ensure that the model has been fit in
    order to perform actions that require a collection of models to have been fit.

    :param fn: Wrapper for methods that require a fit collection of models to be set.
    :return: An instance method
    """

    def _fit_check(self, *args, **kwargs):
        if not self.model:
            raise DivinerException(
                "The model has not been fit. Please fit the model first."
            )
        return fn(self, *args, **kwargs)

    return _fit_check


def _model_init_check(fn):
    """
    Model initialization validation decorator. Ensures that the model hasn't already been fit
    when running certain methods in the GroupedProphet object instance.

    :param fn: Wrapper for methods that require a state where .fit() has not been already executed
               on the instance.
    :return: An instance method
    """

    def model_init_check(self, *args, **kwargs):
        if self.model:
            raise DivinerException(
                "The model has already been fit. Create a new instance to fit the model again."
            )
        return fn(self, *args, **kwargs)

    return model_init_check


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
