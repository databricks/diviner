"""
This Abstract Base Class defines the contract for implementing a specific forecasting library's
grouped modeling application.
"""
import abc
from collections import defaultdict
from typing import Tuple, List
from diviner.exceptions import DivinerException

GROUPED_MODEL_BASE_ATTRIBUTES = ["_group_key_columns", "_master_key"]


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
        self._group_key_columns = None
        self.model = defaultdict(dict)
        self._master_key = None

    @abc.abstractmethod
    def fit(self, df, group_key_columns: Tuple[str], y_col: str, datetime_col: str, **kwargs):
        """
        Generate a grouped representation of the "highly normalized" Dataframe, create a grouping
        key column, and iterate over those grouping keys to generate trained models of the
        plugin-backend time series forecasting library employed in the subclass plugin.
        Plugin implementations must adhere to a data structure of multi-line serializable format
        of ``{grouping_key : <grouping_key_value>, grouping_keys: [<keys>], model: <fit model>}``
        as an instance attribute of the subclass.

        :param df: a Dataframe that contains the keys specified in the class constructor init.
        :param group_key_columns: list of grouping key columns in df that determine unique
                                  time series collections of data.
        :param y_col: column name containing the endogenous regressor used for forecasting.
        :param datetime_col: column name for the datetime values associated with each row of data.
        :param kwargs: Underlying configuration overrides for the per-instance forecasting
                       library used as a model for each group.
        :return: Object instance
        """

    @abc.abstractmethod
    def predict(self, predict_col: str):
        """
        Template method for defining the standard signature for predict.
        This signature requires a DataFrame of the same structure as that used in `.fit()`,
        e.g., a fully normalized structure consisting of the grouping key columns and datetime
        column of the future datetime events to infer predictions for.

        :param predict_col: The column name of the forecasted data to be generated on output.
        :return: A consolidated DataFrame of the union of each model's predictions per each
                 grouping key present in the `df` DataFrame.
        """

    @abc.abstractmethod
    def predict_groups(self, groups: List[Tuple[str]]):
        """
        Template for generating predictions for a subset of the groups that were involved in model
        fitting.
        :param groups: ``List[Tuple[str]]`` the collection of group(s) to generate forecast
                       predictions. The group definitions must be the values within the
                       ``group_key_columns`` that were used during the ``fit`` of the model in
                       order to return valid forecasts.

                       .. Note:: The positional ordering of the values are important and must match
                         the order of ``group_key_columns`` for the ``fit`` argument to provide
                         correct prediction forecasts.

        :return: A consolidated DataFrame of the union of each group model's predictions per each
                 grouping key present in both the ``groups`` argument and the model's fit groups.
        """

    @abc.abstractmethod
    def save(self, path: str):
        """
        Model serialization method, converting the model instance to JSON to be stored on disk.

        :param path: file system path to write the serialized model instance to.
        :return: None
        """

    @abc.abstractmethod
    def load(self, path: str):
        """
        Deserialization method for reading the JSON representation of a model instance of this
        type and initializing an instance of the sub class with the attributes stored during
        a ``save`` method call.

        :param path: The path on the file system that the serialized model is located at.
        :return: An instance of the model, attributes set according to what was in the
                 serialized state of the model.
        """

    def _fit_check(self):
        """
        Model fit validation decorator. Performs a check to ensure that the model has been fit in
        order to perform actions that require a collection of models to have been fit.

        """
        if not self.model:
            raise DivinerException("The model has not been fit. Please fit the model first.")

    def _model_init_check(self):
        """
        Model initialization validation decorator. Ensures that the model hasn't already been fit
        when running certain methods in the GroupedProphet object instance.
        """
        if self.model:
            raise DivinerException(
                "The model has already been fit. Create a new instance to fit the model again."
            )
