import os
from copy import deepcopy
from prophet import Prophet

from diviner.exceptions import DivinerException
from diviner.model.base_model import GroupedForecaster
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.utils.prophet_utils import (
    generate_future_dfs,
    cross_validate_model,
    extract_params,
)
from diviner.utils.common import (
    restructure_fit_payload,
    fit_check,
    model_init_check,
    validate_keys_in_df,
    restructure_predictions,
    create_reporting_df,
)
from diviner.serialize.prophet_serializer import (
    grouped_model_from_json,
    grouped_model_to_json,
)


class GroupedProphet(GroupedForecaster):
    """
    A class for executing multiple Prophet models from a single submitted DataFrame.
    The structure of the input DataFrame to the `.fit()` method must have defined grouping
    columns that are used to build the per-group processing dataframes and models for each group.
    The names of these columns, passed in as part of the `.fit()` method are required to be
    present in the DataFrame schema.
    Any parameters that are needed to override Prophet parameters can be submitted as kwargs
    to the `.fit()` and `.predict()` methods. These settings will be overridden globally for all
    grouped models within the submitted DataFrame.
    """

    def __init__(self, **kwargs):
        """

        :param kwargs: Prophet class configuration overrides
        """
        super().__init__()
        self.prophet_init_kwargs = kwargs
        self.master_key = "grouping_key"

    def _fit_prophet(self, group_key, df, **kwargs):

        return {group_key: Prophet(**self.prophet_init_kwargs).fit(df, **kwargs)}

    @model_init_check
    def fit(self, df, group_key_columns, **kwargs):
        """
        Main fit method for executing a Prophet .fit() on the submitted DataFrame, grouped by
        the `group_key_columns` submitted.
        When initiated, the input DataFrame (`df`) will be split into an iterable collection
        that represents a 'core' series to be fit against.
        :param df: Normalized pandas DataFrame containing group_key_columns, a 'ds' column, and
                   a target 'y' column.
                   An example normalized data set to be used in this method:

                   |region     |zone|ds          |y     |
                   |'northeast'|1   |"2021-10-01"|1234.5|
                   |'northeast'|2   |"2021-10-01"|3255.6|
                   |'northeast'|1   |"2021-10-02"|1255.9|

        :param group_key_columns: The columns in the `df` argument that define, in aggregate, a
                                  unique time series entry. For example, with the DataFrame
                                  referenced in the `df` param, group_key_columns would be:
                                  ('region', 'zone')
        :param kwargs: overrides for underlying Prophet `.fit()` **kwargs (i.e., optimizer backend
                       library configuration overrides)
        :return: object instance
        """

        self.group_key_columns = group_key_columns

        validate_keys_in_df(df, self.group_key_columns)

        grouped_data = PandasGroupGenerator(
            self.group_key_columns
        ).generate_processing_groups(df)

        fit_model = [
            self._fit_prophet(group_key, group_df, **kwargs)
            for group_key, group_df in grouped_data
        ]

        self.model = restructure_fit_payload(fit_model)

        return self

    def _predict_prophet(self, group_key: tuple, df):
        """
        Internal method for predicting a single timeseries group from the specified group_key
        and DataFrame supplied (consisting of, at a minimum, a 'ds' column of datetime events
        to forecast).
        :param group_key: A master_group_key entry tuple generated from the model's fit stage.
        :param df: DataFrame that consists of datetime entries to forecast values for.
        :return: DataFrame consisting of the master_group_key value from the `group_key` argument
                 and the datetime entries and forecasts
        """

        if group_key not in list(self.model.keys()):
            raise DivinerException(
                f"The grouping key '{group_key}' is not in the model instance."
            )
        model = deepcopy(self.model[group_key])
        raw_prediction = model.predict(df)
        raw_prediction.insert(
            0, self.master_key, raw_prediction.apply(lambda x: group_key, axis=1)
        )

        return raw_prediction

    def _run_predictions(self, grouped_data):
        """
        Private method for running predictions for each group in the prediction processing
        collection with a list comprehension
        :param grouped_data: Collection of List[(master_group_key, future_df)]
        :return: A consolidated (unioned) single DataFrame of all groups forecasts
        """
        predictions = [
            self._predict_prophet(group_key, df) for group_key, df in grouped_data
        ]

        return restructure_predictions(
            predictions, self.group_key_columns, self.master_key
        )

    @fit_check
    def predict(self, df):
        """
        Main prediction method for generating forecast values based on the group keys and dates
        for each that are passed in to this method. The structure of the DataFrame submitted to
        this method is the same normalized format that `.fit()` takes as a DataFrame argument.
        i.e.:

        |region     |zone|ds          |
        |'northeast'|1   |"2021-10-01"|
        |'northeast'|2   |"2021-10-01"|
        |'northeast'|1   |"2021-10-02"|

        :param df: Normalized DataFrame consisting of grouping key entries and the dates to
                   forecast for each group.
        :return: A consolidated (unioned) single DataFrame of all groups forecasts
        """

        validate_keys_in_df(df, self.group_key_columns)

        grouped_data = PandasGroupGenerator(
            self.group_key_columns
        ).generate_processing_groups(df)

        return self._run_predictions(grouped_data)

    @fit_check
    def cross_validation(self, horizon, metrics=None, **kwargs):
        """
        Metric scoring method that will run backtesting cross validation scoring for each
        time series specified within the model after a `.fit()` has been performed.
        Default metrics that will be returned:
        `["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]`
        note: If the configuration overrides for the model during `fit()` set
        `uncertainty_samples=0`, the metric `coverage` will be removed from metrics calculation,
        saving a great deal of runtime overhead since the prediction errors (yhat_upper, yhat_lower)
        will not be calculated.
        note: overrides to functionality of both `cross_validation()` and `performance_metrics()`
          within Prophet's `diagnostics` module are handled here as kwargs.
        :param horizon: String pd.Timedelta format that defines the length of forecasting values
                        to generate in order to acquire error metrics.
                        examples: '30 days', '1 year'
        :param metrics: Specific subset list of metrics to calculate and return.
                        note: metrics supplied that are not a member of:
                        `["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]` will
                        raise a Diviner Exception.
                        note: The `coverage` metric will be removed if error estiamtes are not
                        configured to be calculated as part of the Prophet `.fit()` method by
                        setting `uncertainty_samples=0` within the GroupedProphet `.fit()` method.
        :param kwargs: cross validation overrides to Prophet's
                      `prophet.diagnostics.cross_validation()` and
                      `prophet.diagnostics.performance_metrics()` functions
        :return: A consolidated Pandas DataFrame containing the specified metrics
                 to test as columns with each row representing a group.
        """
        scores = {
            group_key: cross_validate_model(model, horizon, metrics, **kwargs)
            for group_key, model in self.model.items()
        }

        return create_reporting_df(scores, self.master_key, self.group_key_columns)

    def extract_model_params(self):
        """
        Utility method for extracting all model parameters from each model within the processed
        groups.
        :return: A consolidated Pandas DataFrame containing the model parameters as columns
                 with each row entry representing a group.
        """

        model_params = {
            group_key: extract_params(model) for group_key, model in self.model.items()
        }
        return create_reporting_df(
            model_params, self.master_key, self.group_key_columns
        )

    @fit_check
    def forecast(self, horizon: int, frequency: str):
        """
        Forecasting method that will automatically generate forecasting values where the 'ds'
        datetime value from the `.fit()` DataFrame left off. For example:
        If the last datetime value in the training data is '2021-01-01 00:01:00', with a
        specified `frequency` of "1 day", the beginning of the forecast value will be
        '2021-01-02 00:01:00' and will continue at a 1 day frequency for `horizon` number of
        entries.
        This implementation wraps the Prophet library's
          `prophet.forecaster.Prophet().make_future_dataframe()` method.

        Note: This will generate a forecast for each group that was present in the
          model `.fit()` input DataFrame. Time horizon values are dependent on the per-group
          'ds' values for each group, which may result in different datetime values if the source
          fit DataFrame did not have consistent datetime values within the 'ds' column for each
          group.

        :param horizon: The number of row events to forecast
        :param frequency: The frequency (periodicity) of Pandas date_range format
                          (i.e., 'D', 'M', 'Y')
        note see for full listing of available strings:
          https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        :return: A consolidated (unioned) single DataFrame of all groups forecasts
        """

        grouped_data = generate_future_dfs(self.model, horizon, frequency)

        return self._run_predictions(grouped_data)

    @fit_check
    def save(self, path: str):
        """
        Serialization of the class instance of this model, provided that it has been fit.
        This will store the model as a JSON string.
        :param path: Location on the file system to store the model artifact.
        :return: None
        """

        directory = os.path.dirname(path)

        if not os.path.exists(directory):
            os.mkdir(directory)

        model_as_json = grouped_model_to_json(self)

        with open(path, "w") as f:
            f.write(model_as_json)

    @model_init_check
    def load(self, path: str):
        """
        Deserialization of the model artifact from local JSON representation to an instance of
        `GroupedProphet()` based on the attributes from a fit model.
        :param path: File system path of a saved GroupedProphet model.
        :return: An instance of GroupedProphet with fit attributes applied.
        """

        if not os.path.isfile(path):
            raise DivinerException(
                f"There is no valid model artifact at the specified path: {path}"
            )
        with open(path, "r") as f:
            raw_model = f.read()

        return grouped_model_from_json(raw_model, "diviner", "GroupedProphet")
