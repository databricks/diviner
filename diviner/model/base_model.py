import abc
from collections import defaultdict
from typing import Tuple
from diviner.config.constants import MASTER_GROUP_KEY

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
        self.master_key = MASTER_GROUP_KEY

    @abc.abstractmethod
    def fit(self, df, group_key_columns: Tuple[str], **kwargs):
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
        """
        Template method for defining the standard signature for predict.
        This signature requires a DataFrame of the same structure as that used in `.fit()`,
        e.g., a fully normalized structure consisting of the grouping key columns and datetime
        column of the future datetime events to infer predictions for.
        :param df: A DataFrame containing grouping key columns that resolve to values that
                   were present in the `.fit()` DataFrame.
        :return: A consolidated DataFrame of the union of each model's predictions per each
                 grouping key present in the `df` DataFrame.
        """
        pass

    @abc.abstractmethod
    def forecast(self, horizon: int, frequency: str):
        """
        A template method that constructs a synthetic period-repeating DataFrame from the
        end period of the training data. The concept here is to not have the user define
        a specific prediction DataFrame, but rather to automatically generate the required
        data from the attributes of the model's training (essentially to 'start predicting
        where the training data left off'.
        :param horizon: int number of future units per group to generate predictions for
        :param frequency: The frequency of the horizon in Pandas time_delta format (e.g.,
               'D' for days, 'M' for month)
        :return: A consolidated DataFrame of the union of each model's predictions for the
                 horizon number of units of frequency.
        """
        pass

    @abc.abstractmethod
    def save(self, path: str):
        """
        Model serialization method, converting the model instance to JSON to be stored on disk.
        :param path: file system path to write the serialized model instance to.
        :return: None
        """
        pass

    @abc.abstractmethod
    def load(self, path: str):
        """
        Deserialization method for reading the JSON representation of a model instance of this
        type and initializing an instance of the sub class with the attributes stored during
        a `save` method call.
        :param path: The path on the file system that the serialized model is located at.
        :return: An instance of the model, attributes set according to what was in the
                 serialized state of the artifact.
        """
        pass
