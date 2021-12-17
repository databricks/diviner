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
)
from diviner.utils.pmdarima_utils import (
    _get_arima_params,
    _get_arima_training_metrics,
    _extract_arima_model,
    _generate_prediction_config,
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

        super().__init__()
        self._y_col = y_col
        self._datetime_col = datetime_col
        self._model_template = model_template
        self._exog_cols = exog_cols
        self._master_key = "grouping_key"
        self._predict_col = predict_col
        self._max_datetime_per_group = None
        self._datetime_freq = None

        #  TODO: record last datetime event from training and period to create datetime mapping
        #  in predict and forecast methods

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


        :param df:
        :param group_key_columns:
        :param silence_warnings:
        :param kwargs:
        :return:
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

        fit_model = [
            self._fit_individual_model(
                group_key, group_df, silence_warnings, **fit_kwargs
            )
            for group_key, group_df in dt_indexed_group_data
        ]

        self.model = _restructure_fit_payload(fit_model)

        return self

    def _predict_single_group(self, row_entry, n_periods_col, exog, **kwargs):
        """

        :param row_entry:
        :return:
        """

        group_key = row_entry[self._master_key]
        return_conf_int = row_entry.get("return_conf_int", False)
        alpha = row_entry.get("alpha", 0.05)
        inverse_transform = row_entry.get("inverse_transform", True)
        model = self._extract_individual_model(group_key)
        prediction = model.predict(
            n_periods=row_entry[n_periods_col],
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

        return prediction_df

    def _predict_groups(self, df, n_periods_col="n_periods", exog=None, **kwargs):
        """
        Predict future periods from the trained models.
        The `df` structure must consist of columns that match the original group_key_columns that
        were passed in during training (the values of combinations will only provide a prediction
        if the group constructed from these grouping columns was present at training). In addition
        to these grouping columns, a column defined as number of future steps to predict must be
        defined. This column will, for each group, contain the per-group future periods to predict
        upon.
        Fields required: group_key_columns(raw), n_periods_col
        Field optional: return_conf_int, alpha, inverse_transform
        :param df:
        :param exog:
        :param kwargs:
        :return:
        """
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
        frequency=None,
        **kwargs,
    ):
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

        :return:
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

        self._fit_check()
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        grouped_pmdarima_save(self, path)

    @classmethod
    def load(cls, path: str):
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
        self, df, group_key_columns, alpha=0.05, test="kpss", max_d=2, **kwargs
    ):
        """

        :param df:
        :param group_key_columns:
        :param alpha:
        :param test:
        :param max_d:
        :param kwargs:
        :return:
        """
        grouped_df = PandasGroupGenerator(group_key_columns).generate_processing_groups(
            df
        )

        group_ndiffs = {
            group: ndiffs(
                x=group_df[self._y_col], alpha=alpha, test=test, max_d=max_d, **kwargs
            )
            for group, group_df in grouped_df
        }

        return group_ndiffs

    def calculate_nsdiffs(
        self, df, group_key_columns, m, test="ocsb", max_D=2, **kwargs
    ):
        """

        :param df:
        :param group_key_columns:
        :param m:
        :param test:
        :param max_D:
        :param kwargs:
        :return:
        """

        grouped_df = PandasGroupGenerator(group_key_columns).generate_processing_groups(
            df
        )
        group_nsdiffs = {
            group: nsdiffs(
                x=group_df[self._y_col], m=m, max_D=max_D, test=test, **kwargs
            )
            for group, group_df in grouped_df
        }

        return group_nsdiffs

    def calculate_is_constant(self, df, group_key_columns):

        grouped_df = PandasGroupGenerator(group_key_columns).generate_processing_groups(
            df
        )
        group_constant_check = {
            group: is_constant(group_df[self._y_col]) for group, group_df in grouped_df
        }
        return group_constant_check

    def cross_validate(self):

        raise NotImplementedError
