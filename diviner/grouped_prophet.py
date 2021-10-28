import os
from copy import deepcopy
from prophet import Prophet
from tqdm import tqdm

from diviner.exceptions import DivinerException
from diviner.model.base_model import GroupedForecaster
from diviner.data.pandas_generator import PandasGroupGenerator
from diviner.config.base_config import BaseConfig
from diviner.config.grouped_prophet.prophet_config import get_prophet_init
from diviner.utils.prophet_utils import (
    generate_future_dfs,
    cross_validate_model,
    create_reporting_df,
    extract_params,
)
from diviner.utils.common import (
    restructure_fit_payload,
    fit_check,
    model_init_check,
    validate_keys_in_df,
    restructure_predictions,
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

    def __init__(self):
        super().__init__()

    @staticmethod
    def _fit_prophet(group_key, df, **model_conf):

        return {group_key: Prophet(**model_conf).fit(df)}

    @model_init_check
    def fit(self, df, group_key_columns, **kwargs):
        """
        Main fit method for executing a Prophet .fit() on the submitted DataFrame, grouped by
        the `group_key_columns` submitted.
        When initiated, the input DataFrame (`df`) will be split into an iterable collection
        that represents a 'core' series to be fit against.
        :param df:
        :param group_key_columns:
        :param kwargs:
        :return:
        """

        self.group_key_columns = group_key_columns

        validate_keys_in_df(df, self.group_key_columns)
        tqdm_on = kwargs.get("tqdm_on", BaseConfig.TQDM_ON)

        grouped_data = PandasGroupGenerator(
            self.group_key_columns
        ).generate_processing_groups(df)

        grouped_data = tqdm(grouped_data) if tqdm_on else grouped_data

        model_conf = get_prophet_init(**kwargs)

        fit_model = [
            self._fit_prophet(group_key, df, **model_conf)
            for group_key, df in grouped_data
        ]

        self.model = restructure_fit_payload(fit_model)

        return self

    def _predict_prophet(self, group_key: tuple, df):

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

    def _run_predictions(self, grouped_data, tqdm_on):

        grouped_data = tqdm(grouped_data) if tqdm_on else grouped_data

        predictions = [
            self._predict_prophet(group_key, df) for group_key, df in grouped_data
        ]

        return restructure_predictions(
            predictions, self.group_key_columns, self.master_key
        )

    @fit_check
    def predict(self, df, **kwargs):

        validate_keys_in_df(df, self.group_key_columns)
        tqdm_on = kwargs.get("tqdm_on", BaseConfig.TQDM_ON)

        grouped_data = PandasGroupGenerator(
            self.group_key_columns
        ).generate_processing_groups(df)

        return self._run_predictions(grouped_data, tqdm_on)

    @fit_check
    def cross_validation(self, cv_initial, cv_period, cv_horizon, parallel=None):

        scores = {
            group_key: cross_validate_model(
                model, cv_initial, cv_period, cv_horizon, parallel
            )
            for group_key, model in self.model.items()
        }

        return create_reporting_df(scores, self.master_key, self.group_key_columns)

    def extract_model_params(self):

        model_params = {
            group_key: extract_params(model) for group_key, model in self.model.items()
        }
        return create_reporting_df(
            model_params, self.master_key, self.group_key_columns
        )

    @fit_check
    def forecast(self, horizon: int, period_type: str, **kwargs):

        tqdm_on = kwargs.get("tqdm_on", BaseConfig.TQDM_ON)
        grouped_data = generate_future_dfs(self.model, horizon, period_type)

        return self._run_predictions(grouped_data, tqdm_on)

    @fit_check
    def save(self, path: str):

        directory = os.path.dirname(path)

        if not os.path.exists(directory):
            os.mkdir(directory)

        model_as_json = grouped_model_to_json(self)

        with open(path, "w") as f:
            f.write(model_as_json)

    @model_init_check
    def load(self, path: str):

        if not os.path.isfile(path):
            raise DivinerException(
                f"There is no valid model artifact at the specified path: {path}"
            )
        with open(path, "r") as f:
            raw_model = f.read()

        return grouped_model_from_json(raw_model, "diviner", "GroupedProphet")
