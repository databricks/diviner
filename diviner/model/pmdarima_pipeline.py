from pmdarima.preprocessing import (
    BoxCoxEndogTransformer,
    LogEndogTransformer,
    FourierFeaturizer,
    DateFeaturizer,
)
from pmdarima.pipeline import Pipeline
from pmdarima.arima import AutoARIMA
from diviner.exceptions import DivinerException


class PmdarimaPipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._key_split = "__"  # Mirrors pipeline kwargs convention in pmdarima package
        self._stage_order = kwargs.get("stage_order", [])

    def _extract_kwargs_keys(self):
        return ", ".join(list(self.kwargs.keys()))

    def _stage_config_extract(self, stage_name):
        try:
            return {
                key.split(self._key_split)[1]: value
                for key, value in self.kwargs.items()
                if key != stage_name and key.startswith(stage_name)
            }
        except IndexError as e:
            raise DivinerException(
                f"Error in extracting configurations for stage '{stage_name}'. "
                "Ensure that stage and argument are separated by 2 "
                f"underscores. {e}"
            )

    def _build_fourier(self):
        conf = self._stage_config_extract("fourier")
        if not conf.get("m"):
            raise DivinerException(
                "Using a FourierFeaturizer requires the key 'fourier__m' to be set. "
                f"Keys passed: {self._extract_kwargs_keys()}"
            )
        return "fourier", FourierFeaturizer(m=conf.pop("m"), **conf)

    def _build_boxcox(self):
        conf = self._stage_config_extract("boxcox")
        return "boxcox", BoxCoxEndogTransformer(**conf)

    def _build_logendog(self):
        conf = self._stage_config_extract("log")
        return "log", LogEndogTransformer(**conf)

    def _build_dates(self):
        conf = self._stage_config_extract("dates")
        if not conf.get("column_name"):
            raise DivinerException(
                "Using a DateFeaturizer requires the key 'dates__column_name' to be set. "
                f"Keys passed: {self._extract_kwargs_keys()}"
            )
        return "dates", DateFeaturizer(column_name=conf.pop("column_name"), **conf)

    def _build_arima(self):
        conf = self._stage_config_extract("arima")
        return "arima", AutoARIMA(**conf)

    def _build_pipeline_stages(self):
        stages = []
        if self.kwargs:
            conf_keys = list(self.kwargs.keys())
            if (
                any(item.startswith("fourier") for item in conf_keys)
                or "fourier" in self._stage_order
            ):
                stages.append(self._build_fourier())
            if (
                any(item.startswith("boxcox") for item in conf_keys)
                or "boxcox" in self._stage_order
            ):
                stages.append(self._build_boxcox())
            if (
                any(item.startswith("log") for item in conf_keys)
                or "log" in self._stage_order
            ):
                stages.append(self._build_logendog())
            if (
                any(item.startswith("dates") for item in conf_keys)
                or "dates" in self._stage_order
            ):
                stages.append(self._build_dates())
        stages.append(self._build_arima())
        return stages

    def _resolve_pipeline_ordering(self, stages):
        ordered_stages = []
        if self._stage_order:
            for stage in self._stage_order:
                ordered_stages.append([item for item in stages if item[0] == stage][0])
            for stage in stages:
                if stage[0] not in self._stage_order:
                    ordered_stages.append(stage)
            if ordered_stages[-1][0] != "arima":
                raise DivinerException(
                    "Ordered preprocessing stages submitted to pipeline must have 'arima' as "
                    f"final stage. Submitted order: '{', '.join(self._stage_order)}'"
                )
        else:
            ordered_stages = stages
        return ordered_stages

    def build_pipeline(self):

        stages = self._build_pipeline_stages()
        resolved_stage_ordering = self._resolve_pipeline_ordering(stages)
        pipeline = Pipeline(resolved_stage_ordering)
        return pipeline
