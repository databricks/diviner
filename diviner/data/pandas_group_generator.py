from diviner.data.base_group_generator import BaseGroupGenerator
from diviner.utils.common import _validate_keys_in_df
import pandas as pd
from typing import Tuple


class PandasGroupGenerator(BaseGroupGenerator):
    """
    This class is used to convert a normalized collection of time series data within a single
    ``DataFrame``, e.g.:

    =========== ==== ============ ======
    region      zone ds           y
    =========== ==== ============ ======
    'northeast' 1    "2021-10-01" 1234.5
    'northeast' 2    "2021-10-01" 3255.6
    'northeast' 1    "2021-10-02" 1255.9
    =========== ==== ============ ======

    With the grouping keys ``['region', 'zone']`` define the unique series of the target ``y``
    indexed by ``ds``.

    This class will

    #. Generate a `master group key` that is a tuple zip of the grouping key arguments specified
       by the user, preserving the order of declaration of these keys.
    #. Group the ``DataFrame`` by these master grouping keys and generate a collection of tuples
       of the form ``(master_grouping_key, <series DataFrame>)`` which is used for iterating over
       to generate the individualized forecasting models for each master key group.

    """

    def __init__(self, group_key_columns: Tuple, datetime_col: str, y_col: str):
        """
        :param group_key_columns: Grouping columns that a combination of which designates a
                                  combination of ``ds`` and ``y`` that represent a distinct series.
        :param datetime_col: The name of the column that contains the ``datetime`` values for
                             each series.
        :param y_col: The endogenous regressor element of the series. This is the value that is
                      used for training and is the element that is intending to be forecast.
        """
        self._group_key_columns = group_key_columns
        self._datetime_col = datetime_col
        self._y_col = y_col
        super().__init__(group_key_columns, datetime_col, y_col)

    def _get_df_with_master_key_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for creating the 'master_group_key' column that defines a unique group.
        The master_group_key column is generated from the concatenation (within a tuple) of the
        values in each of the individual `_group_key_columns`, serving as an aggregation grouping
        key to define a unique collection of datetime series values.
        For example:

        =========== ==== ============ ======
        region      zone ds           y
        =========== ==== ============ ======
        'northeast' 1    "2021-10-01" 1234.5
        'northeast' 2    "2021-10-01" 3255.6
        'northeast' 1    "2021-10-02" 1255.9
        =========== ==== ============ ======

        With the above dataset, the ``group_key_columns`` passed in would be: ``('region', 'zone')``
        This method will modify the input ``DataFrame`` by adding the ``master_group_key`` as
        follows:

        =========== ==== ============ ====== ================
        region      zone ds           y      grouping_key
        =========== ==== ============ ====== ================
        'northeast' 1    "2021-10-01" 1234.5 ('northeast', 1)
        'northeast' 2    "2021-10-01" 3255.6 ('northeast', 2)
        'northeast' 1    "2021-10-02" 1255.9 ('northeast', 1)
        =========== ==== ============ ====== ================

        :param df: The normalized ``DataFrame``
        :return: A copy of the passed-in ``DataFrame`` with a master grouping key column added
                 that contains the group definitions per row of the input ``DataFrame``.
        """

        _validate_keys_in_df(df, self._group_key_columns)

        master_group_df = df.copy()
        master_group_df[self._master_group_key] = master_group_df[[*self._group_key_columns]].apply(
            lambda column: tuple(column), axis=1
        )  # pylint: disable=unnecessary-lambda
        return master_group_df

    def generate_processing_groups(self, df: pd.DataFrame):
        """
        Method for generating the collection of ``[(master_grouping_key, <group DataFrame>)]``

        This method will call ``_create_master_key_column()`` to generate a column containing
        the tuple of the values within the ``_group_key_columns`` fields, then generate an
        iterable collection of ``key`` -> ``DataFrame`` representation.

        For example, after adding the ``grouping_key`` column from ``_create_master_key_column()``,
        the ``DataFrame`` will look like this

        =========== ==== ============ ====== ================
        region      zone ds           y      grouping_key
        =========== ==== ============ ====== ================
        'northeast' 1    "2021-10-01" 1234.5 ('northeast', 1)
        'northeast' 2    "2021-10-01" 3255.6 ('northeast', 2)
        'northeast' 1    "2021-10-02" 1255.9 ('northeast', 1)
        =========== ==== ============ ====== ================

        This method will translate this structure to

        ``[(('northeast', 1),``

          ============ ======
          ds           y
          ============ ======
          "2021-10-01" 1234.5
          "2021-10-02" 1255.9
          ============ ======

          ``),
          (('northeast', 2),``

          ============ ======
          ds           y
          ============ ======
          "2021-10-01" 3255.6
          "2021-10-02" 1255.9
          ============ ======

        ``)]``

        :param df: Normalized ``DataFrame`` that contains the columns defined in instance attribute
                   ``_group_key_columns`` within its schema.
        :return: ``List(tuple(master_group_key, df))`` the processing collection of ``DataFrame``
                 coupled with their group identifier.
        """

        master_key_generation = self._get_df_with_master_key_column(df)

        group_consolidation_df = (
            master_key_generation.groupby([self._master_group_key, self._datetime_col])[self._y_col]
            .agg("sum")
            .reset_index()
        )

        grouped_data = list(
            dict(tuple(group_consolidation_df.groupby(self._master_group_key))).items()
        )

        return grouped_data

    def generate_prediction_groups(self, df: pd.DataFrame):
        """
        Method for generating the data set collection required to run a manual per ``datetime``
        prediction for arbitrary datetime and key groupings.

        :param df: Normalized ``DataFrame`` that contains the columns defined in instance attribute
                   ``_group_key_columns`` within its schema and the dates for prediction within the
                   ``datetime_col`` field.
        :return: ``List(tuple(master_group_key, df))`` the processing collection of
                 ``DataFrame`` coupled with their group identifier.
        """

        master_key_generation = self._get_df_with_master_key_column(df)

        grouped_data = list(
            dict(tuple(master_key_generation.groupby(self._master_group_key))).items()
        )

        return grouped_data
