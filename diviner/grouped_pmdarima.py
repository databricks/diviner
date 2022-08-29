import inspect
import os
import warnings
from copy import deepcopy
from typing import Tuple, List, Dict
from pmdarima import ARIMA, AutoARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.warnings import ModelFitWarning

import numpy as np
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
    _get_datetime_freq_per_group,
)
from diviner.utils.pmdarima_utils import (
    _extract_arima_model,
    _get_arima_params,
    _get_arima_training_metrics,
    _generate_prediction_config,
    _generate_prediction_datetime_series,
    _generate_group_subset_prediction_config,
)
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.data.utils.dataframe_utils import apply_datetime_index_to_groups
from diviner.exceptions import DivinerException


class GroupedPmdarima(GroupedForecaster):
    def __init__(
        self,
        model_template,
    ):
        """
        A class for constructing multiple ``pmdarima`` models from a single normalized input
        DataFrame.
        This implementation supports submission of a model template that is one of:
        ``pmdarima.arima.arima.ARIMA``, ``pmdarima.arima.auto.AutoARIMA``, or
        ``pmdarima.pipeline.Pipeline``.
        The constructor argument of ``model_template`` will apply the settings specified as part of
        instantiation of these classes to all groups within the input DataFrame.

        :param model_template: The type of model to build for each of the groups identified.
                               Supported templates:

                               * ``pmdarima.arima.arima.ARIMA`` - A wrapper around
                               ``statsmodels.api.SARIMAX``.
                               See: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.\
                               arima.ARIMA.html#pmdarima.arima.ARIMA
                               * ``pmdarima.arima.auto.AutoARIMA`` - An auto-tunable order and
                               seasonal order SARIMAX implementation.
                               See: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.\
                               arima.AutoARIMA.html
                               * ``pmdarima.pipeline.Pipeline`` - An sklearn-like pipeline
                               orchestrator for building preprocessing and model components for
                               ``pmdarima``.
                               See: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.\
                               pipeline.Pipeline.html#pmdarima.pipeline.Pipeline

                               For examples showing the usage of each of these template paradigms,
                               see the examples section of this package.


        """
        super().__init__()
        self._y_col = None
        self._datetime_col = None
        self._model_template = model_template
        self._exog_cols = None
        self._master_key = "grouping_key"
        self._predict_col = None
        self._max_datetime_per_group = None
        self._datetime_freq_per_group = None
        self._ndiffs = None
        self._nsdiffs = None

    def _extract_individual_model(self, group_key):

        self._fit_check()
        model_instance = self.model.get(group_key)
        if not model_instance:
            raise DivinerException(f"The model for group {group_key} was not trained.")
        return model_instance

    def _fit_individual_model(self, group_key, group_df, silence_warnings, **fit_kwargs):

        y = group_df[self._y_col]
        model = deepcopy(self._model_template)
        if self._exog_cols:
            exog = group_df[self._exog_cols]
        else:
            exog = None

        # Set 'd' term if pre-calculated with `PmdarimaAnalyzer.calculate_ndiffs`
        if self._ndiffs:
            d_term = self._ndiffs.get(group_key, None)
            if d_term:
                if isinstance(model, ARIMA):
                    setattr(model, "order", (model.order[0], d_term, model.order[2]))
                elif isinstance(model, Pipeline):
                    final_stage = model.steps[-1][1]
                    if isinstance(final_stage, AutoARIMA):
                        setattr(final_stage, "d", d_term)
                    elif isinstance(final_stage, ARIMA):
                        setattr(
                            final_stage,
                            "order",
                            (final_stage.order[0], d_term, final_stage.order[2]),
                        )
                elif isinstance(model, AutoARIMA):
                    setattr(model, "d", d_term)

        # Set 'D' term if pre-calculated with `PmdarimaAnalyzer.calculate_nsdiffs`
        if self._nsdiffs:
            sd_term = self._nsdiffs.get(group_key, None)
            if sd_term:
                if isinstance(model, ARIMA):
                    setattr(
                        model,
                        "seasonal_order",
                        (
                            model.seasonal_order[0],
                            sd_term,
                            model.seasonal_order[2],
                            model.seasonal_order[3],
                        ),
                    )
                elif isinstance(model, Pipeline):
                    final_stage = model.steps[-1][1]
                    if isinstance(final_stage, AutoARIMA):
                        setattr(final_stage, "D", sd_term)
                    elif isinstance(final_stage, ARIMA):
                        setattr(
                            final_stage,
                            "seasonal_order",
                            (
                                final_stage.seasonal_order[0],
                                sd_term,
                                final_stage.seasonal_order[2],
                                final_stage.seasonal_order[3],
                            ),
                        )
                elif isinstance(model, AutoARIMA):
                    setattr(model, "D", sd_term)

        with warnings.catch_warnings():  # Suppress SARIMAX RuntimeWarning
            if silence_warnings:
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=ModelFitWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
            return {group_key: model.fit(y=y, X=exog, **fit_kwargs)}

    def fit(
        self,
        df,
        group_key_columns,
        y_col: str,
        datetime_col: str,
        exog_cols: List[str] = None,
        ndiffs: Dict = None,
        nsdiffs: Dict = None,
        silence_warnings: bool = False,
        **fit_kwargs,
    ):
        """
        Fit method for training a ``pmdarima`` model on the submitted normalized DataFrame.
        When initialized, the input DataFrame will be split into an iterable collection of
        grouped data sets based on the ``group_key_columns`` arguments, which is then used to fit
        individual ``pmdarima`` models (or a supplied ``Pipeline``) upon the templated object
        supplied as a class instance argument `model_template`.
        For API information for ``pmdarima``'s ``ARIMA``, ``AutoARIMA``, and ``Pipeline`` APIs, see:
        https://alkaline-ml.com/pmdarima/modules/classes.html#api-ref

        :param df: A normalized group data set consisting of a datetime column that defines
                   ordering of the series, an endogenous regressor column that specifies the
                   series data for training (e.g. ``y_col``), and column(s) that define the
                   grouping of the series data.

                   An example normalized data set:

                   =========== ===== ======== ============ ======
                   region      zone  country  ds           y
                   =========== ===== ======== ============ ======
                   'northeast' 1     "US"     "2021-10-01" 1234.5
                   'northeast' 2     "US"     "2021-10-01" 3255.6
                   'northeast' 1     "US"     "2021-10-02" 1255.9
                   =========== ===== ======== ============ ======

                   Wherein the grouping_key_columns could be one, some, or all of
                   ``['region', 'zone', 'country']``, the datetime_col would be the `'ds'` column,
                   and the series ``y_col`` (endogenous regressor) would be `'y'`.
        :param group_key_columns: The columns in the ``df`` argument that define, in aggregate, a
                                  unique time series entry. For example, with the DataFrame
                                  referenced in the ``df`` param, group_key_columns could be:
                                  ``('region', 'zone')`` or ``('region')`` or
                                  ``('country', 'region', 'zone')``
        :param y_col: The name of the column within the DataFrame input to any method within this
                      class that contains the endogenous regressor term (the raw data that will
                      be used to train and use as a basis for forecasting).
        :param datetime_col: The name of the column within the DataFrame input that defines the
                             datetime or date values associated with each row of the endogenous
                             regressor (``y_col``) data.
        :param exog_cols: An optional collection of column names within the submitted data to class
                          methods that contain exogenous regressor elements to use as part of model
                          fitting and predicting.

                          Default: ``None``
        :param ndiffs: optional overrides to the ``d`` ``ARIMA`` differencing term for stationarity
                       enforcement.
                       The structure of this argument is a dictionary in the form of:
                       ``{<group_key>: <d_term>}``. To calculate, use
                       ``diviner.PmdarimaAnalyzer.calculate_ndiffs()``

                       Default: ``None``
        :param nsdiffs: optional overrides to the ``D`` SARIMAX seasonal differencing term for
                        seasonal stationarity enforcement.
                        The structure of this argument is a dictionary in the form of:
                        ``{<group_key>: <D_term>}``. To calculate, use
                        :py:meth:``diviner.PmdarimaAnalyzer.calculate_nsdiffs``

                        Default: ``None``
        :param silence_warnings: If ``True``, removes ``SARIMAX`` and underlying optimizer warning
                                 message from stdout printing. With a sufficiently large nubmer of
                                 groups to process, the volume of these messages to stdout may
                                 become very large.

                                 Default: ``False``
        :param fit_kwargs: ``fit_kwargs`` for ``pmdarima``'s ``ARIMA``, ``AutoARIMA``, or
                           ``Pipeline`` stage overrides.
                           For more information, see the ``pmdarima`` docs:
                           https://alkaline-ml.com/pmdarima/index.html
        :return: object instance of ``GroupedPmdarima`` with the persisted fit model attached.
        """

        self._model_init_check()

        self._y_col = y_col
        self._datetime_col = datetime_col
        self._exog_cols = exog_cols
        self._group_key_columns = group_key_columns
        if ndiffs and isinstance(ndiffs, dict):
            self._ndiffs = ndiffs
        if nsdiffs and isinstance(nsdiffs, dict):
            self._nsdiffs = nsdiffs

        _validate_keys_in_df(df, self._group_key_columns)

        grouped_data = PandasGroupGenerator(
            self._group_key_columns, self._datetime_col, self._y_col
        ).generate_processing_groups(df)

        dt_indexed_group_data = apply_datetime_index_to_groups(grouped_data, self._datetime_col)

        self._max_datetime_per_group = _get_last_datetime_per_group(dt_indexed_group_data)
        self._datetime_freq_per_group = _get_datetime_freq_per_group(dt_indexed_group_data)

        fit_model = [
            self._fit_individual_model(group_key, group_df, silence_warnings, **fit_kwargs)
            for group_key, group_df in dt_indexed_group_data
        ]

        self.model = _restructure_fit_payload(fit_model)

        return self

    def _predict_single_group(self, row_entry, n_periods_col, exog, **predict_kwargs):

        group_key = row_entry[self._master_key]
        return_conf_int = row_entry.get("return_conf_int", False)
        alpha = row_entry.get("alpha", 0.05)
        periods = row_entry[n_periods_col]
        inverse_transform = row_entry.get("inverse_transform", True)
        model = self._extract_individual_model(group_key)

        if isinstance(self._model_template, Pipeline):
            prediction = model.predict(
                n_periods=periods,
                X=exog,
                return_conf_int=return_conf_int,
                alpha=alpha,
                inverse_transform=inverse_transform,
                **predict_kwargs,
            )
        else:
            prediction = model.predict(
                n_periods=periods,
                X=exog,
                return_conf_int=return_conf_int,
                alpha=alpha,
                **predict_kwargs,
            )
        if return_conf_int:
            prediction_raw = pd.DataFrame.from_records(prediction).T
            prediction_raw.columns = [self._predict_col, "_yhat_err"]
            prediction_df = pd.DataFrame(
                prediction_raw["_yhat_err"].to_list(),
                columns=["yhat_lower", "yhat_upper"],
            )
            prediction_df.insert(
                loc=0, column=self._predict_col, value=prediction_raw[self._predict_col]
            )
        else:
            prediction_df = pd.DataFrame.from_dict({self._predict_col: prediction})
        prediction_df[self._master_key] = prediction_df.apply(lambda x: group_key, 1)
        prediction_df[self._datetime_col] = _generate_prediction_datetime_series(
            self._max_datetime_per_group.get(group_key),
            self._datetime_freq_per_group.get(group_key),
            periods,
        )

        return prediction_df

    def _run_predictions(self, df, n_periods_col="n_periods", exog=None, **predict_kwargs):

        self._fit_check()
        processing_data = PandasGroupGenerator(
            self._group_key_columns, self._datetime_col, self._y_col
        )._get_df_with_master_key_column(df)

        prediction_collection = [
            self._predict_single_group(row, n_periods_col, exog, **predict_kwargs)
            for idx, row in processing_data.iterrows()
        ]
        return _restructure_predictions(
            prediction_collection, self._group_key_columns, self._master_key
        )

    def predict(
        self,
        n_periods: int,
        predict_col: str = "yhat",
        alpha: float = 0.05,
        return_conf_int: bool = False,
        inverse_transform: bool = True,
        exog=None,
        **predict_kwargs,
    ):
        """
        Prediction method for generating forecasts for each group that has been trained as part of
        a call to ``fit()``.
        Note that ``pmdarima``'s API does not support predictions outside of the defined datetime
        frequency that was validated during training (i.e., if the series endogenous data is at
        an hourly frequency, the generated predictions will be at an hourly frequency and cannot
        be modified from within this method).

        :param n_periods: The number of future periods to generate. The start of the generated
                          predictions will be 1 frequency period after the maximum datetime value
                          per group during training.
                          For example, a data set used for training that has a datetime frequency
                          in days that ends on 7/10/2021 will, with a value of ``n_periods=7``,
                          start its prediction on 7/11/2021 and generate daily predicted values
                          up to and including 7/17/2021.
        :param predict_col: The name to be applied to the column containing predicted data.

                            Default: ``'yhat'``
        :param alpha: Optional value for setting the confidence intervals for error estimates.
                      Note: this is only utilized if ``return_conf_int`` is set to ``True``.

                      Default: ``0.05`` (representing a 95% CI)
        :param return_conf_int: Boolean flag for whether to calculate confidence interval error
                                estimates for predicted values. The intervals of ``yhat_upper`` and
                                ``yhat_lower`` are based on the ``alpha`` parameter.

                                Default: ``False``
        :param inverse_transform: Optional argument used only for ``Pipeline`` models that include
                                  either a ``BoxCoxEndogTransformer`` or a ``LogEndogTransformer``.

                                  Default: ``True``
        :param exog: Exogenous regressor components as a 2-D array.
                     Note: if the model is trained with exogenous regressor components, this
                     argument is required.

                     Default: ``None``
        :param predict_kwargs: Extra ``kwarg`` arguments for any of the transform stages of a
                               ``Pipeline`` or for additional ``predict`` ``kwargs`` to the model
                               instance. ``Pipeline`` ``kwargs`` are specified in the manner of
                               ``sklearn`` ``Pipeline`` format (i.e.,
                               ``<stage_name>__<arg name>=<value>``. e.g., to change the values of
                               a fourier transformer at prediction time, the override would be:
                               ``{'fourier__n_periods': 45})``
        :return: A consolidated (unioned) single DataFrame of predictions per group.
        """
        self._fit_check()
        self._predict_col = predict_col
        prediction_config = _generate_prediction_config(
            self,
            n_periods,
            alpha,
            return_conf_int,
            inverse_transform,
        )
        return self._run_predictions(prediction_config, exog=exog, **predict_kwargs)

    def predict_groups(
        self,
        groups: List[Tuple[str]],
        n_periods: int,
        predict_col: str = "yhat",
        alpha: float = 0.05,
        return_conf_int: bool = False,
        inverse_transform: bool = False,
        exog=None,
        on_error: str = "raise",
        **predict_kwargs,
    ):
        """
        This is a prediction method that allows for generating a subset of forecasts based on the
        collection of keys. By specifying individual groups in the ``groups`` argument, a limited
        scope forecast can be performed without incurring the runtime costs associated with
        predicting all groups.

        :param groups: ``List[Tuple[str]]`` the collection of
                       group (s) to generate forecast predictions. The group definitions must be
                       the values within the ``group_key_columns`` that were used during the
                       ``fit`` of the model in order to return valid forecasts.

                       .. Note:: The positional ordering of the values are important and must match
                         the order of ``group_key_columns`` for the ``fit`` argument to provide
                         correct prediction forecasts.

        :param n_periods: The number of row events to forecast
        :param predict_col: The name of the column in the output ``DataFrame`` that contains the
                            forecasted series data.
                            Default: ``"yhat"``
        :param alpha: Optional value for setting the confidence intervals for error estimates.
                      Note: this is only utilized if ``return_conf_int`` is set to ``True``.

                      Default: ``0.05`` (representing a 95% CI)
        :param return_conf_int: Boolean flag for whether to calculate confidence interval error
                                estimates for predicted values. The intervals of ``yhat_upper`` and
                                ``yhat_lower`` are based on the ``alpha`` parameter.

                                Default: ``False``
        :param inverse_transform: Optional argument used only for ``Pipeline`` models that include
                                  either a ``BoxCoxEndogTransformer`` or a ``LogEndogTransformer``.

                                  Default: ``False``
        :param exog: Exogenous regressor components as a 2-D array.
                     Note: if the model is trained with exogenous regressor components, this
                     argument is required.

                     Default: ``None``
        :param predict_kwargs: Extra ``kwarg`` arguments for any of the transform stages of a
                               ``Pipeline`` or for additional ``predict`` ``kwargs`` to the model
                               instance. ``Pipeline`` ``kwargs`` are specified in the manner of
                               ``sklearn`` ``Pipeline`` format (i.e.,
                               ``<stage_name>__<arg name>=<value>``. e.g., to change the values of
                               a fourier transformer at prediction time, the override would be:
                               ``{'fourier__n_periods': 45})``
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
        self._predict_col = predict_col
        prediction_config = _generate_group_subset_prediction_config(
            self, groups, n_periods, alpha, return_conf_int, inverse_transform, on_error
        )
        return self._run_predictions(prediction_config, exog=exog, **predict_kwargs)

    def get_metrics(self):
        """
        Retrieve the ``ARIMA`` fit metrics that are generated during the ``AutoARIMA`` or
        ``ARIMA`` training event.
        Note: These metrics are not validation metrics. Use the ``cross_validate()`` method for
        retrieving back-testing error metrics.

        :return: ``Pandas`` ``DataFrame`` with metrics provided as columns and a row entry per
                 group.
        """
        self._fit_check()
        metric_extract = {}
        for group in self.model.keys():
            arima_model = _extract_arima_model(self._extract_individual_model(group))
            metric_extract[group] = _get_arima_training_metrics(arima_model)
        return create_reporting_df(metric_extract, self._master_key, self._group_key_columns)

    def get_model_params(self):
        """
        Retrieve the parameters from the ``fit`` ``model_template`` that was passed in and return
        them in a denormalized ``Pandas`` ``DataFrame``. Parameters in the return ``DataFrame``
        are columns with a row for each group defined during ``fit()``.

        :return: ``Pandas`` ``DataFrame`` with ``fit`` parameters for each group.
        """
        self._fit_check()
        params_extract = {}
        for group in self.model.keys():
            arima_model = _extract_arima_model(self._extract_individual_model(group))
            params_extract[group] = _get_arima_params(arima_model)
        return create_reporting_df(params_extract, self._master_key, self._group_key_columns)

    def cross_validate(self, df, metrics, cross_validator, error_score=np.nan, verbosity=0):
        """
        Method for performing cross validation on each group of the fit model.
        The supplied cross_validator to this method will be used to perform either rolling or
        shifting window prediction validation throughout the data set. Windowing behavior for
        the cross validation must be defined and configured through the cross_validator that is
        submitted.
        See: https://alkaline-ml.com/pmdarima/modules/classes.html#cross-validation-split-utilities
        for details on the underlying implementation of cross validation with ``pmdarima``.

        :param df: A ``DataFrame`` that contains the endogenous series and the grouping key columns
                   that were defined during training. Any missing key entries will not be scored.
                   Note that each group defined within the model will be retrieved from this
                   ``DataFrame``. keys that do not exist will raise an Exception.
        :param metrics: A list of metric names or string of single metric name to use for
                        cross validation metric calculation.
        :param cross_validator: A cross validator instance from ``pmdarima.model_selection``
                               (``RollingForecastCV`` or ``SlidingWindowForecastCV``).
                               Note: setting low values of ``h`` or ``step`` will dramatically
                               increase execution time).
        :param error_score: Default value to assign to a score calculation if an error occurs
                            in a given window iteration.

                            Default: ``np.nan`` (a silent ignore of the failure)
        :param verbosity: print verbosity level for ``pmdarima``'s cross validation stages.

                          Default: ``0`` (no printing to stdout)
        :return: ``Pandas DataFrame`` containing the group information and calculated cross
                 validation metrics for each group.
        """

        from diviner.scoring.pmdarima_cross_validate import (
            _cross_validate_grouped_pmdarima,
        )

        self._fit_check()
        group_data = PandasGroupGenerator(
            self._group_key_columns, self._datetime_col, self._y_col
        ).generate_processing_groups(df)

        dt_group_data = apply_datetime_index_to_groups(group_data, self._datetime_col)
        cv_results = _cross_validate_grouped_pmdarima(
            self.model,
            dt_group_data,
            self._y_col,
            metrics,
            cross_validator,
            error_score,
            self._exog_cols,
            verbosity,
        )

        return create_reporting_df(cv_results, self._master_key, self._group_key_columns)

    def save(self, path: str):
        """
        Serialize and write the instance of this class (if it has been fit) to the path specified.
        Note: The serialized model is base64 encoded for top-level items and ``pickle``'d for
        ``pmdarima`` individual group models and any ``Pandas`` ``DataFrame``.

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
        Load a ``GroupedPmdarima`` instance from a saved serialized version.
        Note: This is a class instance and as such, a ``GroupedPmdarima`` instance does not need
        to be initialized in order to load a saved model.
        For example:
        ``loaded_model = GroupedPmdarima.load(<location>)``

        :param path: The path to a serialized instance of ``GroupedPmdarima``
        :return: The ``GroupedPmdarima`` instance that was saved.
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
