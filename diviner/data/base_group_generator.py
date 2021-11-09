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

    def __init__(self, group_key_columns: Tuple):
        """
        Grouping key columns must be defined to serve in the construction of a consolidated
        single unique key that is used to identify a particular unique time series.

        :param group_key_columns: Tuple[str] of column names that determine which elements of the
                                  submitted DataFrame determine uniqueness of a particular
                                  time series.
        """
        if not group_key_columns or len(group_key_columns) == 0:
            raise DivinerException(
                "Argument 'group_key_columns' tuple must contain at "
                "least one string entry."
            )

        self.group_key_columns = group_key_columns
        self.master_group_key = "grouping_key"

    @abc.abstractmethod
    def generate_processing_groups(self, df):
        """
        Abstract method for the generation of processing execution groups for individual models.
        Implementations of this method should generate a processing collection that is a relation
        between the unique combinations of `group_key_columns` values, generated as a
        `master_group_key` entry that defines a specific datetime series for forecasting.

        :param df: The user-input normalized DataFrame with group_key_columns
        :return: An iterable collection of a relation between master_group_key and datetime series
        DataFrame
        """
