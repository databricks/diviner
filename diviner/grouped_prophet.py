import inspect
import os
import json
from ast import literal_eval
from copy import deepcopy
import warnings
from typing import Tuple, List

from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from diviner.exceptions import DivinerException
from diviner.model.base_model import GroupedForecaster, GROUPED_MODEL_BASE_ATTRIBUTES
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.scoring.prophet_cross_validate import (
    group_cross_validation,
    group_performance_metrics,
)
from diviner.utils.prophet_utils import (
    generate_future_dfs,
    _cross_validate_and_score_model,
    _extract_params,
)
from diviner.utils.common import (
    create_reporting_df,
    _restructure_fit_payload,
    _validate_keys_in_df,
    _restructure_predictions,
)


class GroupedProphet(GroupedForecaster):
    """
    A class for executing multiple Prophet models from a single submitted DataFrame.
    The structure of the input DataFrame to the ``fit`` method must have defined grouping
    columns that are used to build the per-group processing dataframes and models for each group.
    The names of these columns, passed in as part of the ``fit`` method are required to be
    present in the DataFrame schema.
    Any parameters that are needed to override Prophet parameters can be submitted as kwargs
    to the ``fit`` and ``predict`` methods. These settings will be overridden globally for all
    grouped models within the submitted DataFrame.

    For the Prophet initialization constructor, showing which arguments are available to be
    passed in as ``kwargs`` in this class constructor, see:
    https://github.com/facebook/prophet/blob/main/python/prophet/forecaster.py
    """

    def __init__(self, **kwargs):
        """
        Constructor for the GroupedProphet class, providing overrides to the underlying ``Prophet``
        model's arguments.

        :param kwargs: Prophet class configuration overrides
        """
        super().__init__()
        self._prophet_init_kwargs = kwargs
        self._master_key = "grouping_key"
        self._datetime_col = "ds"
        self._y_col = "y"

    def _fit_prophet(self, group_key, df, **kwargs):

        try:
            return {group_key: Prophet(**self._prophet_init_kwargs).fit(df, **kwargs)}
        except Exception as e:
            _warning = (
                f"An error occurred while fitting group {group_key}. The group model will not "
                "be included in the model instance. "
                f"Error: {type(e)} Args: {e.args} {e}",
                RuntimeWarning,
            )
            warnings.warn(*_warning, stacklevel=2)
            print(f"WARNING: {_warning[0]}")

    def fit(self, df, group_key_columns, y_col="y", datetime_col="ds", **kwargs):
        """
        Main ``fit`` method for executing a Prophet ``fit`` on the submitted DataFrame, grouped by
        the ``group_key_columns`` submitted.
        When initiated, the input DataFrame ``df`` will be split into an iterable collection
        that represents a core series to be fit against.
        This ``fit`` method is a per-group wrapper around Prophet's ``fit`` implementation. See:
        https://facebook.github.io/prophet/docs/quick_start.html for information on the basic
        API, as well as links to the source code that will demonstrate all of the options
        available for overriding default functionality.
        For a full description of all parameters that are available to the optimizer, run the
        following in a shell:

        .. code-block:: python
            :caption: Retrieving pystan parameters

            import pystan

            help(pystan.StanModel.optimizing)


        :param df: Normalized pandas DataFrame containing ``group_key_columns``, a ``'ds'`` column,
                   and a target ``'y'`` column.
                   An example normalized data set to be used in this method:

                   ========== ==== ============= ======
                   region     zone ds            y
                   ========== ==== ============= ======
                   northeast  1    '2021-10-01'  1234.5
                   northeast  2    '2021-10-01'  3255.6
                   northeast  1    '2021-10-02'  1255.9
                   ========== ==== ============= ======

        :param group_key_columns: The columns in the ``df`` argument that define, in aggregate, a
                                  unique time series entry. For example, with the DataFrame
                                  referenced in the ``df`` param, group_key_columns could be:
                                  (``'region'``, ``'zone'``)
                                  Specifying an incomplete grouping collection, while valid
                                  through this API (i.e., ('region')), can cause serious problems
                                  with any forecast that is built with this API. Ensure that all
                                  relevant keys are defined in the input `df` and declared in this
                                  param to ensure that the appropriate per-univariate series data
                                  is used to train each model.
        :param y_col: The name of the column within the DataFrame input to any method within this
                      class that contains the endogenous regressor term (the raw data that will
                      be used to train and use as a basis for forecasting).
        :param datetime_col: The name of the column within the DataFrame input that defines the
                             datetime or date values associated with each row of the endogenous
                             regressor (``y_col``) data.
        :param kwargs: overrides for underlying ``Prophet`` ``.fit()`` ``**kwargs`` (i.e., optimizer
                       backend library configuration overrides) for further information, see:
                       (https://facebook.github.io/prophet/docs/diagnostics.html\
                       #hyperparameter-tuning).
        :return: object instance (self) of GroupedProphet
        """

        self._model_init_check()
        self._group_key_columns = group_key_columns

        _validate_keys_in_df(df, self._group_key_columns)

        if y_col != "y":
            df.rename(columns={y_col: "y"}, inplace=True)
        if datetime_col != "ds":
            df.rename(columns={datetime_col: "ds"}, inplace=True)

        grouped_data = PandasGroupGenerator(
            self._group_key_columns, self._datetime_col, self._y_col
        ).generate_processing_groups(df)

        fit_model = []
        for group_key, df in grouped_data:
            group_model = self._fit_prophet(group_key, df, **kwargs)
            if group_model:
                fit_model.append(group_model)

        self.model = _restructure_fit_payload(fit_model)

        return self

    def _predict_prophet(self, group_key: tuple, df):
        """
        Internal method for predicting a single time-series group from the specified group_key
        and DataFrame supplied (consisting of, at a minimum, a `'ds'` column of datetime events
        to forecast).

        :param group_key: A master_group_key entry tuple generated from the model's fit stage.
        :param df: DataFrame that consists of datetime entries to forecast values for.
        :return: DataFrame consisting of the master_group_key value from the ``group_key`` argument
                 and the datetime entries and forecasts
        """

        if group_key not in list(self.model.keys()):
            raise DivinerException(f"The grouping key '{group_key}' is not in the model instance.")
        model = deepcopy(self.model.get(group_key))
        raw_prediction = model.predict(df)
        raw_prediction.insert(
            0, self._master_key, raw_prediction.apply(lambda x: group_key, axis=1)
        )

        return raw_prediction

    def _run_predictions(self, grouped_data):
        """
        Private method for running predictions for each group in the prediction processing
        collection with a list comprehension.

        :param grouped_data: Collection of ``List[(master_group_key, future_df)]``
        :return: A consolidated (unioned) single DataFrame of all groups forecasts
        """
        predictions = [self._predict_prophet(group_key, df) for group_key, df in grouped_data]

        return _restructure_predictions(predictions, self._group_key_columns, self._master_key)

    def predict(self, df, predict_col: str = "yhat"):
        """
        Main prediction method for generating forecast values based on the group keys and dates
        for each that are passed in to this method. The structure of the DataFrame submitted to
        this method is the same normalized format that ``fit`` takes as a DataFrame argument.
        i.e.:

        ========== ==== =============
        region     zone ds
        ========== ==== =============
        northeast  1    '2021-10-01'
        northeast  2    '2021-10-01'
        northeast  1    '2021-10-02'
        ========== ==== =============

        :param df: Normalized DataFrame consisting of grouping key entries and the dates to
                   forecast for each group.
        :param predict_col: The name of the column in the output ``DataFrame`` that contains the
                            forecasted series data.
        :return: A consolidated (unioned) single DataFrame of all groups forecasts
        """
        self._fit_check()
        _validate_keys_in_df(df, self._group_key_columns)

        grouped_data = PandasGroupGenerator(
            self._group_key_columns, self._datetime_col, self._y_col
        ).generate_prediction_groups(df)

        predictions = self._run_predictions(grouped_data)

        if predict_col != "yhat":
            predictions.rename(columns={"yhat": predict_col}, inplace=True)

        return predictions

    def predict_groups(
        self,
        groups: List[Tuple[str]],
        horizon: int,
        frequency: str,
        predict_col: str = "yhat",
        on_error: str = "raise",
    ):
        """
        This is a prediction method that allows for generating a subset of forecasts based on the
        collection of keys.

        :param groups: ``List[Tuple[str]]`` the collection of
                       group(s) to generate forecast predictions. The group definitions must be
                       the values within the ``group_key_columns`` that were used during the
                       ``fit`` of the model in order to return valid forecasts.

                       .. Note:: The positional ordering of the values are important and must match
                           the order of ``group_key_columns`` for the ``fit`` argument to provide
                           correct prediction forecasts.

        :param horizon: The number of row events to forecast
        :param frequency: The frequency (periodicity) of Pandas date_range format
                          (i.e., ``'D'``, ``'M'``, ``'Y'``)
        :param predict_col: The name of the column in the output ``DataFrame`` that contains the
                            forecasted series data.
                            Default: ``"yhat"``
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
        :return: A consolidated (unioned) single DataFrame of forecasts for all groups specified
                 in the ``groups`` argument.
        """

        self._fit_check()
        grouped_data = generate_future_dfs(self.model, horizon, frequency, groups, on_error)

        predictions = self._run_predictions(grouped_data)

        if predict_col != "yhat":
            predictions.rename(columns={"yhat": predict_col}, inplace=True)

        return predictions

    def cross_validate(self, horizon, period=None, initial=None, parallel=None, cutoffs=None):
        """
        Utility method for generating the cross validation dataset for each grouping key.
        This is a wrapper around ``prophet.diagnostics.cross_validation`` and uses the
        same signature arguments as that function. It applies each globally to all groups.
        note: the output of this will be a Pandas DataFrame for each grouping key per cutoff
        boundary in the datetime series. The output of this function will be many times larger than
        the original input data utilized for training of the model.

        :param horizon: pandas Timedelta formatted string (i.e. ``'14 days'`` or ``'18 hours'``) to
                        define the amount of time to utilize for a validation set to be created.
        :param period: the periodicity of how often a windowed validation will occur. Default is
                        0.5 * horizon value.
        :param initial: The minimum amount of training data to include in the first cross validation
                        window.
        :param parallel: mode of computing cross validation statistics. One of:
                         (``None``, ``'processes'``, or ``'threads'``)
        :param cutoffs: List of pandas Timestamp values that specify cutoff overrides to be used in
                        conducting cross validation.
        :return: Dictionary of {``'group_key'``: <cross validation Pandas DataFrame>}
        """
        self._fit_check()
        return group_cross_validation(self, horizon, period, initial, parallel, cutoffs)

    def calculate_performance_metrics(
        self, cv_results, metrics=None, rolling_window=0.1, monthly=False
    ):
        """
            Model debugging utility function for evaluating performance metrics from the grouped
            cross validation extract.
            This will output a metric table for each time boundary from cross validation, for
            each model.
            note: This output will be very large and is intended to be used as a debugging
            tool only.

            :param cv_results: The output return of ``group_cross_validation``
            :param metrics: (Optional) overrides (subtractive) for metrics to generate for this
                            function's output.
                            note: see supported metrics in Prophet documentation:
                            (https://facebook.github.io/prophet/docs/diagnostics.\
                            html#cross-validation)
                            note: any model in the collection that was fit with the argument
                            ``uncertainty_samples`` set to ``0`` will have the metric ``'coverage'``
                            removed from evaluation due to the fact that ```yhat_error``` values are
                            not calculated with that configuration of that parameter.
            :param rolling_window: Defines how much data to use in each rolling window as a
                                   range of ``[0, 1]`` for computing the performance metrics.
            :param monthly: If set to true, will collate the windows to ensure that horizons
                            are computed as number of months from the cutoff date. Only useful
                            for date data that has yearly seasonality associated with calendar
                            day of month.
            :return: Dictionary of {``'group_key'``: <performance metrics per window pandas
                     DataFrame>}
            """
        self._fit_check()
        return group_performance_metrics(cv_results, self, metrics, rolling_window, monthly)

    def cross_validate_and_score(
        self,
        horizon,
        period=None,
        initial=None,
        parallel=None,
        cutoffs=None,
        metrics=None,
        **kwargs,
    ):
        """
        Metric scoring method that will run backtesting cross validation scoring for each
        time series specified within the model after a ``fit`` has been performed.

        Note: If the configuration overrides for the model during ``fit`` set
        ``uncertainty_samples=0``, the metric ``coverage`` will be removed from metrics calculation,
        saving a great deal of runtime overhead since the prediction errors
        ``(yhat_upper, yhat_lower)`` will not be calculated.

        Note: overrides to functionality of both ``cross_validation`` and ``performance_metrics``
        within Prophet's ``diagnostics`` module are handled here as ``kwargs``.
        These arguments in this method's signature are directly passed, per model, to prophet's
        ``cross_validation`` function.

        :param horizon: String pandas ``Timedelta`` format that defines the length of forecasting
                        values to generate in order to acquire error metrics.
                        examples: ``'30 days'``, ``'1 year'``
        :param metrics: Specific subset list of metrics to calculate and return.
                        note: see supported metrics in Prophet documentation:
                        https://facebook.github.io/prophet/docs/diagnostics.html#cross-validation
                        note: The ``coverage`` metric will be removed if error estiamtes are not
                        configured to be calculated as part of the Prophet ``fit`` method by
                        setting ``uncertainty_samples=0`` within the GroupedProphet ``fit`` method.
        :param period: the periodicity of how often a windowed validation will occur. Default is
                       0.5 * horizon value.
        :param initial: The minimum amount of training data to include in the first cross validation
                        window.
        :param parallel: mode of computing cross validation statistics.
                         Supported modes: (``None``, ``'processes'``, or ``'threads'``)
        :param cutoffs: List of pandas ``Timestamp`` values that specify cutoff overrides to be
                        used in conducting cross validation.
        :param kwargs: cross validation overrides to Prophet's
                       ``prophet.diagnostics.cross_validation`` and
                       ``prophet.diagnostics.performance_metrics`` functions
        :return: A consolidated Pandas DataFrame containing the specified metrics
                 to test as columns with each row representing a group.
        """

        self._fit_check()
        scores = {
            group_key: _cross_validate_and_score_model(
                model, horizon, period, initial, parallel, cutoffs, metrics, **kwargs
            )
            for group_key, model in self.model.items()
        }

        return create_reporting_df(scores, self._master_key, self._group_key_columns)

    def extract_model_params(self):
        """
        Utility method for extracting all model parameters from each model within the processed
        groups.

        :return: A consolidated pandas DataFrame containing the model parameters as columns
                 with each row entry representing a group.
        """

        self._fit_check()
        model_params = {
            group_key: _extract_params(model) for group_key, model in self.model.items()
        }
        return create_reporting_df(model_params, self._master_key, self._group_key_columns)

    def forecast(self, horizon: int, frequency: str):
        """
        Forecasting method that will automatically generate forecasting values where the ``'ds'``
        datetime value from the ``fit`` DataFrame left off. For example:
        If the last datetime value in the training data is ``'2021-01-01 00:01:00'``, with a
        specified ``frequency`` of ``'1 day'``, the beginning of the forecast value will be
        ``'2021-01-02 00:01:00'`` and will continue at a 1 day frequency for ``horizon`` number of
        entries.
        This implementation wraps the Prophet library's
        ``prophet.forecaster.Prophet.make_future_dataframe`` method.

        Note: This will generate a forecast for each group that was present in the
        ``fit`` input DataFrame ``df`` argument. Time horizon values are dependent on the per-group
        ``'ds'`` values for each group, which may result in different datetime values if the source
        fit DataFrame did not have consistent datetime values within the ``'ds'`` column for each
        group.

        Note: For full listing of supported periodicity strings for the ``frequency`` parameter,
        see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        :param horizon: The number of row events to forecast
        :param frequency: The frequency (periodicity) of Pandas date_range format
                          (i.e., ``'D'``, ``'M'``, ``'Y'``)
        :return: A consolidated (unioned) single DataFrame of forecasts for all groups
        """

        self._fit_check()
        grouped_data = generate_future_dfs(self.model, horizon, frequency)

        return self._run_predictions(grouped_data)

    def save(self, path: str):
        """
        Serialize the model as a JSON string and write it to the provided path.
        note: The model must be fit in order to save it.

        :param path: Location on the file system to store the model.
        :return: None
        """
        self._fit_check()
        directory = os.path.dirname(path)

        os.makedirs(directory, exist_ok=True)

        model_as_json = self._grouped_model_to_json()

        with open(path, "w") as f:
            f.write(model_as_json)

    @classmethod
    def load(cls, path: str):
        """
        Load the model from the specified path, deserializing it from its JSON string
        representation and returning a ``GroupedProphet`` instance.

        :param path: File system path of a saved ``GroupedProphet`` model.
        :return: An instance of GroupedProphet with ``fit`` attributes applied.
        """

        attr_dict = cls._grouped_model_from_json(path)
        init_args = inspect.signature(cls.__init__).parameters.keys()
        init_cls = [attr_dict[arg] for arg in init_args if arg not in {"self", "kwargs"}]
        instance = cls(*init_cls)
        for key, value in attr_dict.items():
            if key not in init_args:
                setattr(instance, key, value)

        return instance

    def _grouped_model_to_json(self):
        """
        Serialization helper to convert a ``GroupedProphet`` instance to JSON for saving to disk.

        :return: serialized json string of the model's attributes
        """

        model_dict = self._grouped_model_to_dict()
        for key in vars(self).keys():
            if key != "model":
                model_dict[key] = getattr(self, key)

        return json.dumps(model_dict)

    def _grouped_model_to_dict(self):

        model_dict = {attr: getattr(self, attr) for attr in GROUPED_MODEL_BASE_ATTRIBUTES}
        model_dict["model"] = {
            str(master_key): model_to_json(model) for master_key, model in self.model.items()
        }
        return model_dict

    @classmethod
    def _grouped_model_from_dict(cls, raw_model):

        deser_model_payload = {
            literal_eval(master_key): model_from_json(payload)
            for master_key, payload in raw_model.items()
        }
        return deser_model_payload

    @classmethod
    def _grouped_model_from_json(cls, path):
        """
        Helper function to load the grouped model structure from serialized JSON and deserialize
        the ``Prophet`` instances.

        :param path: The storage location of a saved ``GroupedProphet`` object
        :return: Dictionary of instance attributes
        """
        if not os.path.isfile(path):
            raise DivinerException(f"There is no valid model saved at the specified path: {path}")
        with open(path, "r") as f:
            raw_model = json.load(f)

        model_deser = cls._grouped_model_from_dict(raw_model["model"])
        raw_model["model"] = model_deser

        return raw_model
