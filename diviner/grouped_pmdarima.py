import inspect
import os
import warnings
from copy import deepcopy
from pmdarima.arima import decompose, ndiffs, nsdiffs, is_constant
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from diviner.model.base_model import GroupedForecaster
from diviner.serialize.pmdarima_serializer import (
    grouped_pmdarima_save,
    grouped_pmdarima_load,
)
from diviner.utils.common import (
    _validate_keys_in_df,
    _restructure_fit_payload,
    _restructure_predictions,
    create_reporting_df,
    _get_last_datetime_per_group,
    _get_datetime_freq_per_group
)
from diviner.utils.pmdarima_utils import (
    _get_arima_params,
    _get_arima_training_metrics,
    _extract_arima_model,
    _generate_prediction_config,
    _generate_prediction_datetime_series
)
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.data.utils.dataframe_utils import apply_datetime_index_to_groups
from diviner.exceptions import DivinerException


class GroupedPmdarima(GroupedForecaster):

    def __init__(
        self,
        y_col,
        datetime_col,
        model_template,
        exog_cols=None,
        predict_col="forecast",
    ):
        """
        A class for constructing multiple pmdarima models from a single normalized input DataFrame.
        This implementation supports submission of a model template that is one of:
        `pmdarima.arima.arima.ARIMA`, `pmdarima.arima.auto.AutoARIMA`, or
        `pmdarima.pipeline.Pipeline`.
        The constructor argument of `model_template` will apply the settings specified as part of
        instantiation of these classes to all groups within the input DataFrame.

        :param y_col: The name of the column within the DataFrame input to any method within this
                      class that contains the endogenous regressor term (the 'raw data' that will
                      be used to train and use as a basis for forecasting).
        :param datetime_col: The name of the column within the DataFrame input that defines the
                             datetime or date values associated with each row of the endogenous
                             regressor (y_col) data.
        :param model_template: The model template to be applied to each of the groups identified.
                               Supported templates:
                               `pmdarima.arima.arima.ARIMA` - A wrapper around
                               `statsmodels.api.SARIMAX`.
                               See: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.\
                               arima.ARIMA.html#pmdarima.arima.ARIMA
                               `pmdarima.arima.auto.AutoARIMA` - An auto-tunable order and seasonal
                               order SARIMAX implementation.
                               See: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.\
                               arima.AutoARIMA.html
                               `pmdarima.pipeline.Pipeline` - An sklearn-like pipeline orchestrator
                               for building preprocessing and model components for pmdarima.
                               See: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.\
                               pipeline.Pipeline.html#pmdarima.pipeline.Pipeline
                               For examples showing the usage of each of these template paradigms,
                               see the examples section of this package.
        :param exog_cols: Optional exogenous regressor elements to use as part of model fitting
                          and predicting.
        :param predict_col: The name to be applied to the column containing predicted data.
        """
        super().__init__()
        self._y_col = y_col
        self._datetime_col = datetime_col
        self._model_template = model_template
        self._exog_cols = exog_cols
        self._master_key = "grouping_key"
        self._predict_col = predict_col
        self._max_datetime_per_group = None
        self._datetime_freq_per_group = None

    def _model_type_resolver(self, obj=None):

        if not obj:
            obj = self._model_template
        try:
            module = inspect.getmodule(obj).__name__
            clazz = type(obj).__name__
            return module, clazz
        except AttributeError:
            return None, None

    def _extract_individual_model(self, group_key):

        self._fit_check()
        model_instance = self.model.get(group_key)
        if not model_instance:
            raise DivinerException(f"The model for group {group_key} was not trained.")
        return model_instance

    def _fit_individual_model(
        self, group_key, group_df, silence_warnings, **fit_kwargs
    ):

        y = group_df[self._y_col]
        model = deepcopy(self._model_template)
        if self._exog_cols:
            exog = group_df[self._exog_cols]
        else:
            exog = None
        with warnings.catch_warnings():  # Suppress SARIMAX RuntimeWarning
            if silence_warnings:
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return {group_key: model.fit(y=y, X=exog, **fit_kwargs)}

    def fit(self, df, group_key_columns, silence_warnings=False, **fit_kwargs):
        """
        Fit method for training a pmdarima model on the submitted normalized DataFrame.
        When initialized, the input DataFrame will be split into an iterable collection of
        grouped data sets based on the `group_key_columns` arguments, which is then used to fit
        individual pmdarima models (or a supplied Pipeline) upon the templated object supplied
        as a class instance argument `model_template`.
        For API information for pmdarima's ARIMA, AutoARIMA, and Pipeline APIs, see:
        https://alkaline-ml.com/pmdarima/modules/classes.html#api-ref

        :param df: A normalized group data set consisting of a datetime column that defines
                   ordering of the series, an endogenous regressor column that specifies the
                   series data for training (e.g. `y_col`), and column(s) that define the grouping
                   of the series data.
                   An example normalized data set:

                   |region     |zone|country |ds          |y     |
                   |'northeast'|1   |"US"    |"2021-10-01"|1234.5|
                   |'northeast'|2   |"US"    |"2021-10-01"|3255.6|
                   |'northeast'|1   |"US"    |"2021-10-02"|1255.9|

                   Wherein the grouping_key_columns could be one, some, or all of
                   ['region', 'zone', 'country'], the datetime_col would be the 'ds' column, and
                   the series `y_col` (endogenous regressor) would be 'y'.
        :param group_key_columns: The columns in the `df` argument that define, in aggregate, a
                                  unique time series entry. For example, with the DataFrame
                                  referenced in the `df` param, group_key_columns could be:
                                  ('region', 'zone') or ('region') or ('country', 'region', 'zone')
        :param silence_warnings: If True, removes SARIMAX and underlying optimizer warning message
                                 from stdout printing. With a sufficiently large nubmer of groups
                                 to process, the volume of these messages to stdout may become
                                 very large.
        :param kwargs: fit_kwargs for pmdarima's ARIMA, AutoARIMA, or Pipeline stages overrides.
                       For more information, see pmdarima docs:
                       https://alkaline-ml.com/pmdarima/index.html
        :return: object instance of GroupedPmdarima with the persisted fit model attached.
        """

        #  TODO: allow for passed-in per-group 'd' and 'D' values from ndiffs and nsdiffs

        self._model_init_check()
        self._group_key_columns = group_key_columns

        _validate_keys_in_df(df, self._group_key_columns)

        grouped_data = PandasGroupGenerator(
            self._group_key_columns
        ).generate_processing_groups(df)

        dt_indexed_group_data = apply_datetime_index_to_groups(
            grouped_data, self._datetime_col
        )

        self._max_datetime_per_group = _get_last_datetime_per_group(dt_indexed_group_data)
        self._datetime_freq_per_group = _get_datetime_freq_per_group(dt_indexed_group_data)

        fit_model = [
            self._fit_individual_model(
                group_key, group_df, silence_warnings, **fit_kwargs
            )
            for group_key, group_df in dt_indexed_group_data
        ]

        self.model = _restructure_fit_payload(fit_model)

        return self

    def _predict_single_group(self, row_entry, n_periods_col, exog, **kwargs):

        group_key = row_entry[self._master_key]
        return_conf_int = row_entry.get("return_conf_int", False)
        alpha = row_entry.get("alpha", 0.05)
        periods = row_entry[n_periods_col]
        inverse_transform = row_entry.get("inverse_transform", True)
        model = self._extract_individual_model(group_key)
        prediction = model.predict(
            n_periods=periods,
            X=exog,
            return_conf_int=return_conf_int,
            alpha=alpha,
            inverse_transform=inverse_transform,
            **kwargs,
        )
        if return_conf_int:
            prediction_raw = pd.DataFrame.from_records(prediction).T
            prediction_raw.columns = [self._predict_col, "yhat"]
            prediction_df = pd.DataFrame(
                prediction_raw["yhat"].to_list(), columns=["yhat_lower", "yhat_upper"]
            )
            prediction_df.insert(
                loc=0, column=self._predict_col, value=prediction_raw[self._predict_col]
            )
        else:
            prediction_df = pd.DataFrame.from_dict({self._predict_col: prediction})
        prediction_df[self._master_key] = prediction_df.apply(lambda x: group_key, 1)
        prediction_df[self._datetime_col] = _generate_prediction_datetime_series(self._max_datetime_per_group.get(group_key), self._datetime_freq_per_group.get(group_key), periods)

        return prediction_df

    def _predict_groups(self, df, n_periods_col="n_periods", exog=None, **kwargs):

        self._fit_check()
        processing_data = PandasGroupGenerator(
            self._group_key_columns
        )._get_df_with_master_key_column(df)

        prediction_collection = [
            self._predict_single_group(row, n_periods_col, exog, **kwargs)
            for idx, row in processing_data.iterrows()
        ]
        return _restructure_predictions(
            prediction_collection, self._group_key_columns, self._master_key
        )

    def predict(
        self,
        n_periods: int,
        alpha=0.05,
        return_conf_int=False,
        inverse_transform=True,
        exog=None,
        **kwargs,
    ):
        """
        Prediction method for generating forecasts for each group that has been trained as part of
        a call to `fit()`.
        Note that pmdarima's API does not support predictions outside of the defined datetime
        frequency that was validated during training (i.e., if the series endogenous data is at
        an hourly frequency, the generated predictions will be at an hourly frequency and cannot
        be modified from within this method).

        :param n_periods: The number of future periods to generate. The start of the generated
                          predictions will be 1 frequency period after the maximum datetime value
                          per group during training.
                          For example, a data set used for training that has a datetime frequency
                          in days that ends on 7/10/2021 will, with a value of `n_periods=7`,
                          start its prediction on 7/11/2021 and generate daily predicted values
                          up to and including 7/17/2021.
        :param alpha: Optional value for setting the confidence intervals for error estimates.
                      Note: this is only utilized if `return_conf_int` is set to `True`.
                      Default: 0.05 (representing a 95% CI)
        :param return_conf_int: Boolean flag for whether to calculate confidence interval error
                                estimates for predicted values. The intervals of `yhat_upper` and
                                `yhat_lower` are based on the `alpha` parameter.
                                Default: False
        :param inverse_transform: Optional argument used only for Pipeline models that include
                                  either a `BoxCoxEndogTransformer` or a `LogEndogTransformer`.
                                  Default: True
        :param exog: Exogenous regressor components as a 2-D array.
                     Note: if the model is trained with exogenous regressor components, this
                     argument is required.
        :param kwargs: Extra kwarg arguments for any of the transform stages of a Pipelie or for
                       additional `predict` kwargs to the model instance.
                       Pipeline kwargs are specified in the manner of sklearn Pipelines (i.e.,
                       <stage_name>__<arg name>=<value>. e.g., to change the values of a fourier
                       transformer at prediction time, the override would be:
                       {'fourier__n_periods': 45})
        :return: A consolidate (unioned) single DataFrame of predictions per group.
        """
        self._fit_check()
        prediction_config = _generate_prediction_config(
            self,
            n_periods,
            alpha,
            return_conf_int,
            inverse_transform,
        )
        return self._predict_groups(prediction_config, exog=exog, **kwargs)

    def get_metrics(self):
        """
        Retrieve the ARIMA fit metrics that are generated during the AutoARIMA training event.
        These metrics are not validation metrics.

        :return: Pandas DataFrame with metrics provided as columns and a row entry per group.
        """
        self._fit_check()
        metric_extract = {}
        for group, pipeline in self.model.items():
            arima_model = _extract_arima_model(self._extract_individual_model(group))
            metric_extract[group] = _get_arima_training_metrics(arima_model)
        return create_reporting_df(
            metric_extract, self._master_key, self._group_key_columns
        )

    def get_model_params(self):
        """
        Retrieve the parameters from the fit `model_template` that was passed in as a Pandas
        DataFrame. Parameters will be columns with a row for each group defined during `fit`.

        :return: Pandas DataFrame with fit parameters for each group.
        """
        #  TODO: extract params from pipeline stages and submit in report
        self._fit_check()
        params_extract = {}
        for group, pipeline in self.model.items():
            arima_model = _extract_arima_model(self._extract_individual_model(group))
            params_extract[group] = _get_arima_params(arima_model)
        return create_reporting_df(
            params_extract, self._master_key, self._group_key_columns
        )

    def save(self, path: str):
        """
        Serialize and write the instance of this class (if it has been fit) to the path specified.
        Note: The serialized model is base64 encoded for top-level items and pickle'd for
        pmdarima individual group models and any Pandas DataFrame.

        :param path: Path to write this model's instance to.
        :return: None
        """
        self._fit_check()
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        grouped_pmdarima_save(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a GroupedPmdarima instance from a saved serialized version.
        Note: This is a class instance and as such, a GroupedPmdarima instance does not need to be
        initialized in order to load a saved model.
        For example:
        `loaded_model = GroupedPmdarima.load(<location>)`

        :param path: The path to a serialized instance of GroupedPmdarima
        :return: The GroupedPmdarima instance that was saved.
        """
        attr_dict = grouped_pmdarima_load(path)
        init_args = inspect.signature(cls.__init__).parameters.keys()
        cleaned_attr_dict = {key.lstrip("_"): value for key, value in attr_dict.items()}
        init_cls = [cleaned_attr_dict[arg] for arg in init_args if arg != "self"]
        instance = cls(*init_cls)
        for key, value in attr_dict.items():
            if key not in init_args:
                setattr(instance, key, value)
        return instance

    def _decompose_group(self, group_df, group_key, m, type_, filter_):

        group_decomposition = decompose(
            x=group_df[self._y_col], type_=type_, m=m, filter_=filter_
        )
        group_result = {
            key: getattr(group_decomposition, key)
            for key in group_decomposition._fields
        }
        output_df = pd.DataFrame.from_dict(group_result)
        output_df[self._datetime_col] = group_df[self._datetime_col]
        output_df[self._master_key] = output_df.apply(lambda x: group_key, 1)
        return output_df

    def decompose_groups(self, df, group_key_columns, m, type_, filter_=None):
        """
        Utility method that wraps `pmdarima.arima.decompose()` for each group within the
        passed-in DataFrame.
        Note: decomposition works best if the total number of entries within the series being
        decomposed is a multiple of the `m` parameter value.

        :param df: A normalized group-defined data set from which to decompose the elements of
                   the endogenous series into columns: {'trend', 'seasonal', and 'random'} that
                   represent the primary components of the series.
        :param group_key_columns: The columns in the `df` argument that define, in aggregate, a
                                  unique time series entry.
        :param m: The frequency of the endogenous series. (i.e., for daily data, m might be '7'
                  for estimating a weekly seasonality, or '365' for yearly seasonality effects.)
        :param type_: The type of decomposition to perform. One of: ['additive', 'multiplicative']
                      See:
        :param filter_: Optional Array for performing convolution. This is specified as a
                        filter for coefficients (the Moving Average and/or
                        Auto Regressor coefficients) in reverse time order in order to filter out
                        a seasonal component.
                        Default: None
        :return: Pandas DataFrame with the decomposed trends for each group.
        """
        grouped_df = PandasGroupGenerator(group_key_columns).generate_processing_groups(
            df
        )
        group_decomposition = {
            group_key: self._decompose_group(group_df, group_key, m, type_, filter_)
            for group_key, group_df in grouped_df
        }
        return _restructure_predictions(
            group_decomposition, group_key_columns, self._master_key
        )

    def calculate_ndiffs(
        self, df, group_key_columns, alpha=0.05, test="kpss", max_d=2
    ):
        """
        Utility method for determining the optimal `d` value for ARIMA ordering. Calculating this
        as a fixed value can dramatically increase the tuning time for pmdarima.

        :param df: A normalized group-defined data set from which to calculate per-group optimized
                   values of the `d` component of the ARIMA ordering of `('p', 'd', 'q')`
        :param group_key_columns: The columns in the `df` argument that define, in aggregate, a
                                  unique time series entry.
        :param alpha: significance level for determining if a pvalue used for testing a
                      value of 'd' is significant or not.
                      Default: 0.05
        :param test: Type of unit test for stationarity determination to use.
                     Supported values: ['kpss', 'adf', 'pp']
                     See:
                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.KPSSTest.\
                     html#pmdarima.arima.KPSSTest
                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.\
                     html#pmdarima.arima.PPTest
                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ADFTest.\
                     html#pmdarima.arima.ADFTest
                     Default: 'kpss'
        :param max_d: The max value for `d` to test.
        :return: Dictionary of {<group_key>: <optimal 'd' value>}
        """
        grouped_df = PandasGroupGenerator(group_key_columns).generate_processing_groups(
            df
        )

        group_ndiffs = {
            group: ndiffs(
                x=group_df[self._y_col], alpha=alpha, test=test, max_d=max_d
            )
            for group, group_df in grouped_df
        }

        return group_ndiffs

    def calculate_nsdiffs(
        self, df, group_key_columns, m, test="ocsb", max_D=2
    ):
        """
        Utility method for determining the optimal `D` value for seasonal SARIMAX ordering.

        :param df: A normalized group-defined data set from which to calculate per-group optimized
                   values of the `D` component of the SARIMAX seasonal ordering of
                   `('P', 'D', 'Q', 's')`
        :param group_key_columns: The columns in the `df` argument that define, in aggregate, a
                                  unique time series entry.
        :param m: The number of seasonal periods in the series.
        :param test: Type of unit test for seasonality.
                     Supported tests: ['ocsb', 'ch']
                     See:
                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.\
                     html#pmdarima.arima.OCSBTest
                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.\
                     html#pmdarima.arima.CHTest
                     Default: 'ocsb'
        :param max_D: Maximum number of seasonal differences to test for.
                      Default: 2
        :return: Dictionary of {<group_key>: <optimal 'D' value>}
        """

        grouped_df = PandasGroupGenerator(group_key_columns).generate_processing_groups(
            df
        )
        group_nsdiffs = {
            group: nsdiffs(
                x=group_df[self._y_col], m=m, max_D=max_D, test=test
            )
            for group, group_df in grouped_df
        }

        return group_nsdiffs

    def calculate_is_constant(self, df, group_key_columns):
        """
        Utility method for determining whether or not a series is composed of all of the same
        elements or not. (e.g. a series of {1, 2, 3, 4, 5, 1, 2, 3} will return 'False', while
        a series of {1, 1, 1, 1, 1, 1, 1, 1, 1} will return 'True')

        :param df: A normalized group-defined data set from which to calculate constancy
        :param group_key_columns: The columns in the `df` argument that define, in aggregate, a
                                  unique time series entry.
        :return: Dictionary of {<group_key>: <Boolean constancy check>}
        """
        grouped_df = PandasGroupGenerator(group_key_columns).generate_processing_groups(
            df
        )
        group_constant_check = {
            group: is_constant(group_df[self._y_col]) for group, group_df in grouped_df
        }
        return group_constant_check

    def cross_validate(self):

        raise NotImplementedError
