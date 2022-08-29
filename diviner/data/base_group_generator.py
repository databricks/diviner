"""
Abstract Base Class for defining the API contract for group generator operations.
This base class is a template for package-specific implementations that function to
convert a normalized representation of grouped time series into per-group collections
of discrete time series so that forecasting models can be trained on each group.
"""
import abc
from typing import Tuple
from diviner.exceptions import DivinerException


class BaseGroupGenerator(abc.ABC):
    """
    Abstract class for defining the basic elements of performing a group processing collection
    generation operation.
    """

    def __init__(self, group_key_columns: Tuple, datetime_col: str, y_col: str):
        """
        Grouping key columns must be defined to serve in the construction of a consolidated
        single unique key that is used to identify a particular unique time series. The
        unique combinations of these provided fields define and control the grouping of
        univariate series data in order to train (fit) a particular model upon each of the
        unique series (that are defined by the combination of the values within these supplied
        columns).

        The primary purpose of the children of this class is to generate a dictionary of:
        ``{<group_key> : <DataFrame with unique univariate series>}``.
        The ```group_key`` element is constructed as a tuple of the values within the columns
        specified by ``_group_key_columns`` in this class constructor.

        For example, with a normalized data set provided of:

        =========== ==== ====== ======
        ds          y    group1 group2
        =========== ==== ====== ======
        2021-09-02  11.1 "a"    "z"
        2021-09-03  7.33 "a"    "z"
        2021-09-02  31.1 "b"    "q"
        2021-09-03  44.1 "b"    "q"
        =========== ==== ====== ======

        There are two separate univariate series: ``("a", "z")`` and ``("b", "q")``.
        The group generator's function is to convert this unioned ``DataFrame`` into the following:

        ``{ ("a", "z"):``

        ========== ==== ====== ======
        ds         y    group1 group2
        ========== ==== ====== ======
        2021-09-02 11.1 "a"    "z"
        2021-09-03 7.33 "a"    "z"
        ========== ==== ====== ======

          ``,("b", "q"):``

        ========== ==== ====== ======
        ds         y    group1 group2
        ========== ==== ====== ======
        2021-09-02 31.1 "b"    "q"
        2021-09-03 44.1 "b"    "q"
        ========== ==== ====== ======

        ``}``

        This grouping allows for a model to be fit to each of these series in isolation.

        :param group_key_columns: ``Tuple[str]`` of column names that determine which elements of
                                  the submitted ``DataFrame`` determine uniqueness of a particular
                                  time series.
        :param datetime_col: The name of the column that contains the ``datetime`` values for
                             each series.
        :param y_col: The endogenous regressor element of the series. This is the value that is
                      used for training and is the element that is intending to be forecast.
        """
        if not group_key_columns or len(group_key_columns) == 0:
            raise DivinerException(
                "Argument '_group_key_columns' tuple must contain at " "least one string entry."
            )

        self._group_key_columns = group_key_columns
        self._datetime_col = datetime_col
        self._y_col = y_col
        self._master_group_key = "grouping_key"

    @abc.abstractmethod
    def generate_processing_groups(self, df):
        """
        Abstract method for the generation of processing execution groups for individual models.
        Implementations of this method should generate a processing collection that is a relation
        between the unique combinations of ``_group_key_columns`` values, generated as a
        ``_master_group_key`` entry that defines a specific datetime series for forecasting.

        For example, with a normalized dataframe input of

        ========== ======= ======= ==
        ds         region  country y
        ========== ======= ======= ==
        2020-01-01 SW      USA     42
        2020-01-02 SW      USA     11
        2020-01-01 NE      USA     31
        2020-01-01 Ontario CA      12
        ========== ======= ======= ==

        The output structure should be, with the group_keys value specified as:

        ``("country", "region"):[{ ("USA", "SW"):``

        ========== ====== ======= ==
        ds         region country y
        ========== ====== ======= ==
        2020-01-01 SW     USA     42
        2020-01-02 SW     USA     11
        ========== ====== ======= ==

        ``}. {("USA", "NE"):``

        ========== ====== ======= ==
        ds         region country y
        ========== ====== ======= ==
        2020-01-01 NE     USA     31
        ========== ====== ======= ==

        ``}, {("CA", "Ontario"):``

        ========== ======= ======= ==
        ds         region  country y
        ========== ======= ======= ==
        2020-01-01 Ontario CA      12
        ========== ======= ======= ==

        ``}]``

        The list wrapper around dictionaries is to allow for multiprocessing support without having
        to contend with encapsulating the entire dictionary for the processing of a single key
        and value pair.

        :param df: The user-input normalized DataFrame with _group_key_columns
        :return: A list of dictionaries of ``{group_key: <group's univariate series data>}``
                 structure for isolated processing by the model APIs.
        """

    @abc.abstractmethod
    def generate_prediction_groups(self, df):
        """
        Abstract method for generating the data set collection required for manual prediction for
        arbitrary datetime and key groupings.

        :param df: Normalized ``DataFrame`` that contains the columns defined in instance attribute
                   ``_group_key_columns`` within its schema and the dates for prediction within the
                   ``datetime_col`` field.
        :return: ``List(tuple(master_group_key, df))`` the processing collection of ``DataFrame``
                 coupled with their group identifier.
        """
