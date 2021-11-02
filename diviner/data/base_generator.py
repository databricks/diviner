import abc
from typing import List


class BaseGroupGenerator(abc.ABC):
    """
    Abstract class for defining the basic elements of performing a group processing collection
    generation operation.
    """
    def __init__(self, group_key_columns: List[str]):
        """
        Grouping key columns must be defined to serve as the basis for constructing a 'meta column'
        that is used for performing iterative model creation and forecasting.
        :param group_key_columns: List[str] of column names that determine which elements of the
                                  submitted DataFrame determine uniqueness of a particular
                                  time series.
        """
        self.group_key_columns = group_key_columns
        self.master_group_key = "grouping_key"

    @abc.abstractmethod
    def generate_processing_groups(self, df):
        pass
