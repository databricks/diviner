import pandas as pd

from diviner.model.base_model import GroupedForecaster
from diviner.config.grouped_statsmodels.statsmodels_config import (
    get_statsmodels_model,
    extract_fit_kwargs,
)
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.data.utils.dataframe_utils import apply_datetime_index_to_groups
from diviner.utils.common import (
    validate_keys_in_df,
    validate_prediction_config_df,
    restructure_fit_payload,
    generate_forecast_horizon_series,
    convert_forecast_horizon_series_to_df,
    restructure_predictions,
)
from diviner.scoring.statsmodels_scoring import extract_statsmodels_metrics
from diviner.utils.common import create_reporting_df
from diviner.utils.statsmodels_utils import _get_max_datetime_per_group
from diviner.config.constants import PREDICT_END_COL, PREDICT_START_COL


class GroupedStatsmodels(GroupedForecaster):
    def __init__(
        self,
        model_type: str,
        endog_column: str,
        time_col: str,
        exog_column=None,
        predict_col="forecast",
    ):
        super().__init__()
        self.model_clazz = get_statsmodels_model(model_type)
        self.endog_column = endog_column
        self.time_col = time_col
        self.exog_column = exog_column
        self.max_datetime_per_group = None
        self.predict_col = predict_col

    def _fit_model(self, group_key, df, **kwargs):

        endog = df[self.endog_column]

        kwarg_extract = extract_fit_kwargs(self.model_clazz, **kwargs)

        if self.exog_column:
            model = self.model_clazz(endog, df[self.exog_column], **kwarg_extract.clazz)
        else:
            model = self.model_clazz(endog, **kwarg_extract.clazz)

        return {group_key: model.fit(**kwarg_extract.fit)}

    def fit(self, df, group_key_columns, **kwargs):

        self.group_key_columns = group_key_columns

        validate_keys_in_df(df, self.group_key_columns)

        grouped_data = PandasGroupGenerator(
            self.group_key_columns
        ).generate_processing_groups(df)

        dt_indexed_group_data = apply_datetime_index_to_groups(
            grouped_data, self.time_col
        )

        self.max_datetime_per_group = _get_max_datetime_per_group(dt_indexed_group_data)

        fit_model = [
            self._fit_model(group_key, group_df, **kwargs)
            for group_key, group_df in dt_indexed_group_data
        ]

        self.model = restructure_fit_payload(fit_model)

        return self

    def _predict_single_group(self, row_entry):
        group_key = row_entry[self.master_key]
        model = self.model[group_key]
        start = row_entry[PREDICT_START_COL]
        end = row_entry[PREDICT_END_COL]
        prediction = pd.DataFrame(
            model.predict(start=start, end=end), columns=[self.predict_col]
        )
        prediction.index.name = self.time_col
        prediction = prediction.reset_index()
        prediction[self.master_key] = prediction.apply(lambda x: group_key, 1)
        return prediction

    def predict(self, df):

        validate_prediction_config_df(df, self.group_key_columns)

        processing_data = PandasGroupGenerator(
            self.group_key_columns
        )._create_master_key_column(df)

        prediction_collection = [
            self._predict_single_group(row) for idx, row in processing_data.iterrows()
        ]

        return restructure_predictions(
            prediction_collection, self.group_key_columns, self.master_key
        )

    def score_model(self, metrics=None, warning=False):
        """

        :param metrics:
        :param warning: Whether to capture warnings to logs (False) or to print warnings to stdout
                        (True). Default: False
        :return: A Pandas DataFrame consisting of a row per model key group and metrics columns
                 that are available as extracted attributes from the model type used.
                 note: Not all model implementations return all metric types.
        """

        metric_extract = extract_statsmodels_metrics(self.model, metrics, warning)
        return create_reporting_df(
            metric_extract, self.master_key, self.group_key_columns
        )

    def forecast(self, horizon: int, frequency: str = None):

        group_forecast_series_boundaries = generate_forecast_horizon_series(
            self.max_datetime_per_group, horizon, frequency
        )

        group_prediction_collection = convert_forecast_horizon_series_to_df(
            group_forecast_series_boundaries, self.group_key_columns
        )

        return self.predict(group_prediction_collection)

    def save(self, path: str):

        # The save artifact implementation has to deal with the 'Results' instance of the model
        # type and not the original model type.
        # Need to specify the result type through inspection so that when serde of the artifact
        # it can be loaded as the correct type (maintain a lookup dict for model_type: result_type)
        pass

    def load(self, path: str):
        pass
