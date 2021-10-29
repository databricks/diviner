import abc
from collections import defaultdict


class GroupedForecaster(abc.ABC):
    """
    Base class exposing consistent public methods for any plugin grouped forecasting API.

    Implementors of this base class that serve as plugins to this architecture must
    override and conform to these method signatures when creating these plugins for different
    underlying time series forecasting libraries.

    .. Note::
        Subclasses must raise :py:class:`diviner.exceptions.DivinerException` in error cases
        that involve the logical execution involved in this package.
    """

    def __init__(self):
        self.group_key_columns = None
        self.model = defaultdict(dict)
        self.master_key = "grouping_key"

    @abc.abstractmethod
    def fit(self, df, group_key_columns, **kwargs):
        """
        Generate a grouped representation of the "highly normalized" Dataframe, create a grouping
        key column, and iterate over those grouping keys to generate trained models of the
        plugin-backend time series forecasting library employed in the subclass plugin.
        Plugin implementations must adhere to a data structure of multi-line serializable format
        of {grouping_key : <grouping_key_value>, grouping_keys: [<keys>], model: <fit model>}
        as an instance attribute of the subclass.

        :param df: a Dataframe that contains the keys specified in the class constructor init.
        :param group_key_columns: list of grouping key columns in df that determine unique
                                  time series collections of data.
        :param kwargs: Underlying configuration overrides for the per-instance forecasting
                       library used as a model for each group.
        :return: Object instance
        """
        pass

    @abc.abstractmethod
    def predict(self, df):
        pass

    @abc.abstractmethod
    def forecast(self, horizon: int, frequency: str):
        pass

    @abc.abstractmethod
    def save(self, path: str):
        pass

    @abc.abstractmethod
    def load(self, path: str):
        pass
