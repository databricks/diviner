from diviner.model.base_model import GroupedForecaster
from diviner.utils.common import (
    _validate_keys_in_df,
    _restructure_fit_payload,
    create_reporting_df,
)
from diviner.utils.pmdarima_utils import (
    _get_arima_params,
    _get_arima_training_metrics,
    _extract_arima_model_from_pipeline,
)
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.data.utils.dataframe_utils import apply_datetime_index_to_groups
from diviner.model.pmdarima_pipeline import PmdarimaPipeline
from diviner.exceptions import DivinerException


class GroupedPmdarima(GroupedForecaster):
    def __init__(self, y_col, time_col, exog_cols=None, **kwargs):

        super().__init__()
        self._y_col = y_col
        self._time_col = time_col
        self._exog_cols = exog_cols
        self._pmdarima_kwargs = kwargs
        self._master_key = "grouping_key"

    def _extract_individual_pipeline(self, group_key):

        self._fit_check()
        model_instance = self.model.get(group_key)
        if not model_instance:
            raise DivinerException(f"The model for group {group_key} was not trained.")
        return model_instance

    def _fit_individual_model(self, group_key, group_df, **kwargs):

        pipeline = PmdarimaPipeline(**kwargs).build_pipeline()
        y = group_df[self._y_col]
        if self._exog_cols:
            exog = group_df[self._exog_cols]
        else:
            exog = None
        return {group_key: pipeline.fit(y=y, X=exog)}

    def fit(self, df, group_key_columns, **kwargs):
        """

        :param df:
        :param group_key_columns:
        :param kwargs:
        :return:
        """

        self._model_init_check()
        self._group_key_columns = group_key_columns

        _validate_keys_in_df(df, self._group_key_columns)

        grouped_data = PandasGroupGenerator(
            self._group_key_columns
        ).generate_processing_groups(df)
        dt_indexed_group_data = apply_datetime_index_to_groups(
            grouped_data, self._time_col
        )

        fit_model = [
            self._fit_individual_model(group_key, group_df, **kwargs)
            for group_key, group_df in dt_indexed_group_data
        ]

        self.model = _restructure_fit_payload(fit_model)

        return self

    def predict(self, df):

        raise NotImplementedError

    def forecast(self, horizon: int, frequency: str):

        raise NotImplementedError

    def get_metrics(self):
        """
        Retrieve the ARIMA fit metrics that are generated during the AutoARIMA training event.
        These metrics are not validation metrics.
        :return: Pandas DataFrame with metrics provided as columns and a row entry per group.
        """
        self._fit_check()
        metric_extract = {}
        for group, pipeline in self.model.items():
            arima_model = _extract_arima_model_from_pipeline(
                self._extract_individual_pipeline(group)
            )
            metric_extract[group] = _get_arima_training_metrics(arima_model)
        return create_reporting_df(
            metric_extract, self._master_key, self._group_key_columns
        )

    def get_model_params(self):
        """

        :return:
        """
        self._fit_check()
        params_extract = {}
        for group, pipeline in self.model.items():
            arima_model = _extract_arima_model_from_pipeline(
                self._extract_individual_pipeline(group)
            )
            params_extract[group] = _get_arima_params(arima_model)
        return create_reporting_df(
            params_extract, self._master_key, self._group_key_columns
        )

    def save(self, path: str):

        raise NotImplementedError

    def load(self, path: str):

        raise NotImplementedError

    def validate_series(self, df, m, **kwargs):

        # TODO: ADFTest, CHTest, KPSSTest, OCSBTest, PPTest in a single report
        raise NotImplementedError

    def calculate_ndiffs(self, df):

        raise NotImplementedError

    def calculate_nsdiffs(self, df):

        raise NotImplementedError

    def calculate_is_constant(self, df):

        raise NotImplementedError
