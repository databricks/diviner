from diviner.model.base_model import GroupedForecaster


class GroupedPmdarima(GroupedForecaster):

    def __init__(self, **kwargs):

        super().__init__()
        self._pmdarima_kwargs = kwargs
        self._master_key = "grouping_key"

    #  TODO: pipeline constructor

    def _fit_individual_model(self):
        raise NotImplementedError

    def fit(self, group_key_columns, **kwargs):

        raise NotImplementedError

    def predict(self, df):

        raise NotImplementedError

    def forecast(self, horizon: int, frequency: str):

        raise NotImplementedError

    def save(self, path: str):

        raise NotImplementedError

    def load(self, path: str):

        raise NotImplementedError