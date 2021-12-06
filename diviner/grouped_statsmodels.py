import pandas as pd
import inspect
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import warnings

from diviner.model.base_model import GroupedForecaster
from diviner.config.grouped_statsmodels.statsmodels_config import (
    _get_statsmodels_model,
    _extract_fit_kwargs,
)
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.data.utils.dataframe_utils import apply_datetime_index_to_groups
from diviner.utils.common import (
    _validate_keys_in_df,
    validate_prediction_config_df,
    _restructure_fit_payload,
    generate_forecast_horizon_series,
    convert_forecast_horizon_series_to_df,
    _restructure_predictions,
)
from diviner.scoring.statsmodels_scoring import _extract_statsmodels_metrics
from diviner.utils.common import create_reporting_df
from diviner.utils.statsmodels_utils import (
    _get_max_datetime_per_group,
    _resolve_forecast_duration_var_model,
    _extract_params_from_model,
)
from diviner.config.constants import PREDICT_END_COL, PREDICT_START_COL
from diviner.serialize.statsmodels_serializer import (
    group_statsmodels_save,
    group_statsmodels_load,
)


class GroupedStatsmodels(GroupedForecaster):
    """
    Grouped model for processing a grouped collection of a single type of statsmodels model upon
    a single dataset.
    """

    def __init__(
        self,
        model_type: str,
        endog,
        time_col: str,
        exog_column=None,
        predict_col="forecast",
        suppress_logs=True,
    ):
        """
        :param model_type: String value of statsmodels model type to use. Examples:
            "Holt", "ARIMA", "ExponentialSmoothing", "VAR"
            For a full listing, see: https://www.statsmodels.org/stable/tsa.html#statespace-models
            note: not all models are supported. Diagnostic model types that do not have a `.fit()`
            implementation or require a prior fit model as a dependency are not supported.
            The models that are supported are:
            'AR', 'ARDL', 'ARIMA', 'ArmaProcess', 'AutoReg', 'DynamicFactor', 'DynamicFactorMQ',
            'ETSModel', 'ExponentialSmoothing', 'Holt', 'MarkovAutoregression',
            'MarkovRegression', 'SARIMAX', 'STL', 'STLForecast', 'SVAR', 'SimpleExpSmoothing',
            'UECM', 'UnobservedComponents', 'VAR', 'VARMAX', 'VECM'
        :param endog: The endogenous regression component(s) of the observed time series process
            (also known as the 'y' variable(s)). This can be supplied as a single string to denote
            a column for a univariate model or a list of strings of column names for use in
            multivariate models.
        :param time_col: The datetime column of the input dataset that denotes the temporal
            component. Note: statsmodels does not allow for non datetime-indexed series to be
            submitted. Internally, a datetime index will be applied based on the column provided
            in this argument.
        :param exog_column: Optional exogenous regressor components for models that support this.
        :param predict_col: The name of the column to generate predictions (forecasts) as.
            note: this is used in the methods `.forecast()` and `.predict()`.
        """
        super().__init__()
        self.model_type = model_type
        self.model_clazz = _get_statsmodels_model(model_type)
        self.endog = endog
        self.time_col = time_col
        self.exog_column = exog_column
        self.max_datetime_per_group = None
        self.predict_col = predict_col
        self.suppress_logs = suppress_logs
        if self.suppress_logs:
            warnings.filterwarnings("ignore")

    @staticmethod
    @contextmanager
    def suppress_fit_warnings():
        with open(os.devnull, "w") as null_file:
            with redirect_stdout(null_file) as stdout, redirect_stderr(
                null_file
            ) as stderr:
                yield stdout, stderr

    def _fit_model(self, group_key, df, **kwargs):
        """
        Internal method for fitting an individual model based on the group extract data
        :param group_key: The grouping key (master_key) that defines the individual series
        :param df: The subset of data that defines a single series from a group's key
        :param kwargs: Pass-through overrides for the underlying model type
        :return: Dictionary of {group_key: model}
        """

        endog = df[self.endog]

        kwarg_extract = _extract_fit_kwargs(self.model_clazz, **kwargs)

        if self.exog_column:
            model = self.model_clazz(endog, df[self.exog_column], **kwarg_extract.clazz)
        else:
            model = self.model_clazz(endog, **kwarg_extract.clazz)

        if (
            self.suppress_logs
            and "disp" in inspect.signature(model.fit).parameters.keys()
        ):
            kwarg_extract.fit["disp"] = False

        if self.suppress_logs:
            with self.suppress_fit_warnings():
                return {group_key: model.fit(**kwarg_extract.fit)}
        else:
            return {group_key: model.fit(**kwarg_extract.fit)}

    def fit(self, df, group_key_columns, **kwargs):
        """
        Fits a model for each group defined in the `group_key_columns` of the provided `df`
        normalized dataset.
        :param df: Normalized pandas DataFrame containing group_key_columns, a 'time_col` column,
                   and a endogenous column (optionally multiple columns for certain types of
                   models). If `exog_columns` is specified in the class constructor, this column
                   must be present as well.
                   An example normalized data set to be used in this method:

                   |region     |zone|ds          |endog |
                   |'northeast'|1   |"2021-10-01"|1234.5|
                   |'northeast'|2   |"2021-10-01"|3255.6|
                   |'northeast'|1   |"2021-10-02"|1255.9|
        :param group_key_columns: tuple of column names to be used for identifying unique time
                                  series data.
                                  The columns in the `df` argument that define, in aggregate, a
                                  unique time series entry. For example, with the DataFrame
                                  referenced in the `df` param, group_key_columns could be:
                                  ('region', 'zone') or ('zone', 'region')
        :param kwargs: overrides to the underlying model being used to fit the data against.
                        For full reference to underlying supported statsmodels models and their
                        available overrides, see: https://www.statsmodels.org/stable/tsa.html
        :return: self
        """
        self._model_init_check()
        self._group_key_columns = group_key_columns

        _validate_keys_in_df(df, self._group_key_columns)

        grouped_data = PandasGroupGenerator(
            self._group_key_columns
        ).generate_processing_groups(df)

        dt_indexed_group_data = apply_datetime_index_to_groups(
            grouped_data, self.time_col
        )

        self.max_datetime_per_group = _get_max_datetime_per_group(dt_indexed_group_data)

        fit_model = [
            self._fit_model(group_key, group_df, **kwargs)
            for group_key, group_df in dt_indexed_group_data
        ]

        self.model = _restructure_fit_payload(fit_model)

        return self

    def _predict_single_group(self, row_entry):
        """
        Internal method for generating the prediction for a given group's model based on the
        passed-in predict `df` structure.
        :param row_entry: A row from the `df` argument in `predict()`, representing a single
            group's model.
        :return: A Pandas DataFrame of predictions for an individual group.
        """
        group_key = row_entry[self._master_key]
        model = self.model[group_key]
        start = row_entry[PREDICT_START_COL]
        end = row_entry[PREDICT_END_COL]
        if self.model_type == "VAR":
            units = _resolve_forecast_duration_var_model(row_entry)
            raw_prediction = pd.DataFrame(model.forecast(model.endog, units))
        else:
            raw_prediction = pd.DataFrame(model.predict(start=start, end=end))
        prediction_name = raw_prediction.columns[0]
        prediction = raw_prediction.rename({prediction_name: self.predict_col}, axis=1)
        prediction.index.name = self.time_col
        prediction = prediction.reset_index()
        prediction[self._master_key] = prediction.apply(lambda x: group_key, 1)
        return prediction

    def predict(self, df):
        """
        Method for generating prediction data for each group within the model.
        The input structure for this method's `df` argument is similar to the training data used
        in fitting in that each grouping key column is specified (and must be present in the
        training data for the model). For each group defined in the grouping columns, a start
        datetime and end datetime must be provided for each group.
        As an example:
        Training DF:
        |region    |country    |ds          |y
        |north     |CA         |2019-01-01  |1345
        |south     |CA         |2019-01-01  |3339
        |north     |CA         |2019-01-02  |2223

        To utilize this method, the prediction df would be:
        |region   |country |start      |end
        |"north"  |"CA"    |2020-01-02 |2020-04-01
        |"south"  |"CA"    |2020-01-01 |2020-05-01
        Unlike the `forecast` method, within-training forecasts can be performed with this method.
        Frequency of the training data set will be used to set the datetime distance between the
        provided start and end values in the `df` argument. (i.e., if the frequency of the training
        data was hourly, and the start and end periods are in days, predictions with
        hour-periodicity between the start and end values will be generated).
        :param df: DataFrame consisting of group key entries (one row per group) and the start and
            end time to generate predictions between.
        :return: A unioned DataFrame that contains the predictions for each group.
        """

        self._fit_check()
        validate_prediction_config_df(df, self._group_key_columns)

        processing_data = PandasGroupGenerator(
            self._group_key_columns
        )._get_df_with_master_key_column(df)

        prediction_collection = [
            self._predict_single_group(row) for idx, row in processing_data.iterrows()
        ]

        return _restructure_predictions(
            prediction_collection, self._group_key_columns, self._master_key
        )

    def get_metrics(self, metrics=None, warning=False):
        """
        Retrieves the fit metrics from each grouped model and restructures them into a Pandas
        DataFrame with a single row per group and columns defined by the available metrics
        from the underlying model type used in the class instance.
        :param metrics: A restrictive subset of metrics to extract from each group's fit model
            results object. Due to implementation details in different models, not all metrics
            are available to all models. By default, the following metrics will be extracted:
            `{"aic", "aicc", "bic", "hqic", "mae", "mse", "sse"}`
            note: these are the values extracted from the statsmodels instance of the model and
            are not derived through backtesting or cross validation.
        :param warning: Whether to capture warnings to logs (False) or to print warnings to stdout
                        (True). Default: False
        :return: A Pandas DataFrame consisting of a row per model key group and metrics columns
                 that are available as extracted attributes from the model type used.
                 note: Not all model implementations return all metric types.
        """

        self._fit_check()
        metric_extract = _extract_statsmodels_metrics(self.model, metrics, warning)
        return create_reporting_df(
            metric_extract, self._master_key, self._group_key_columns
        )

    def get_model_params(self):
        """
        Extracts the fitted parameters from the model. This will only extract each model's
        results `param` attribute. For all other attributes associated with a fit model,
        these can be accessed from the `model` attribute after fitting.
        :return: A Pandas DataFrame consisting of one row per group model with columns defined
            as members of the `.params` attribute from the underlying model results type.
        """

        self._fit_check()
        model_params = {
            group_key: _extract_params_from_model(model)
            for group_key, model in self.model.items()
        }
        return create_reporting_df(
            model_params, self._master_key, self._group_key_columns
        )

    def forecast(self, horizon: int, frequency: str = None):
        """
        Generates a forecast of predicted values that starts at the next frequency point after
        the end of the training data set.
        For example, if the training data consisted of date values as follows:
        |ds         |y
        |2021-01-01 |123
        |2021-01-02 |143
        |2021-01-03 |184

        Then the beginning of the forecast output would be 2021-01-04 by default.
        The horizon value will specify the number of future forecast points to generate.
        In the example above, if `horizon` is set to 14, 2 weeks of per-day forecast predictions
        will be generated starting on 2021-01-04 and ending on 2021-01-17.
        :param horizon: The number of time period elements to generate predictive forecasts for,
            starting at the next time period after the end of the training series data.
        :param frequency: (optional) the frequency in Pandas date_range format (.e.g., "D" for day)
        note: for a full listing of available and supported frequencies, see:
          https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        :return: A unioned Pandas DataFrame containing group keys, datetime indexes, and
            the forecasted values starting at the next time period from the end of each group's
            training data.
        """

        self._fit_check()
        group_forecast_series_boundaries = generate_forecast_horizon_series(
            self.max_datetime_per_group, horizon, frequency
        )

        group_prediction_collection = convert_forecast_horizon_series_to_df(
            group_forecast_series_boundaries, self._group_key_columns
        )

        return self.predict(group_prediction_collection)

    def save(self, path: str):
        """
        Save an instance of GroupedStatsmodels to storage. The internal type of each group's model
        will be reprsented as a string-encoded byte array from the internal pickle method from
        each model ResultsWrapper implementation. The model will be stored as a JSON string.
        :param path: Location to write the fitted model to.
        :return: None
        """

        self._fit_check()
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)

        group_statsmodels_save(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a saved GroupedStatmodels model from storage.
        :param path: Location of a saved GroupedStatsmodels instance.
        return: instance of GroupedStatsmodels, deserialized from the saved state.
        """

        attr_dict = group_statsmodels_load(path)
        init_args = inspect.signature(cls.__init__).parameters.keys()
        init_cls = [attr_dict[arg] for arg in init_args if arg != "self"]
        instance = cls(*init_cls)
        for key, value in attr_dict.items():
            if key not in init_args:
                setattr(instance, key, value)

        return instance
