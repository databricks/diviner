from diviner.model.base_model import GroupedForecaster
from diviner.config.grouped_statsmodels.statsmodels_config import get_statsmodels_model
from diviner.data.pandas_generator import PandasGroupGenerator
from diviner.utils.common import validate_keys_in_df, restructure_fit_payload


class GroupedStatsmodels(GroupedForecaster):
    def __init__(self, model_type: str, endog_column: str, exog_column=None):
        super().__init__()
        self.model_clazz = get_statsmodels_model(model_type)
        self.endog_column = endog_column
        self.exog_column = exog_column

    def _fit_model(self, group_key, df, *args, **kwargs):

        endog = df[self.endog_column]

        if self.exog_column:
            model = self.model_clazz(endog, df[self.exog_column], *args, **kwargs)
        else:
            model = self.model_clazz(endog, *args, **kwargs)

        return {group_key: model.fit()}

    def fit(self, df, group_key_columns, *args, **kwargs):

        self.group_key_columns = group_key_columns

        validate_keys_in_df(df, self.group_key_columns)

        grouped_data = PandasGroupGenerator(
            self.group_key_columns
        ).generate_processing_groups(df)

        fit_model = [
            self._fit_model(group_key, group_df, *args, **kwargs)
            for group_key, group_df in grouped_data
        ]

        self.model =  restructure_fit_payload(fit_model)

        return self

    def predict(self, df):
        pass

    def forecast(self, horizon: int, frequency: str):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass