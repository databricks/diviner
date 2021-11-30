import pickle
import json
from ast import literal_eval

from diviner.v1.config.grouped_statsmodels.statsmodels_config import (
    _get_statsmodels_model,
)

_SERDE_EXCLUDE = {"model_clazz"}
_SERDE_PICKLE_OBJ = {"model", "max_datetime_per_group"}


class ModelEncoder:
    def __init__(self):
        self.serialized_model = {}

    def _encode_obj(self, obj):
        group_serialized = {}
        for group_key, val in obj.items():
            group_serialized[str(group_key)] = str(pickle.dumps(val))
        return group_serialized

    def encode(self, grouped_instance):
        model_encode = {}
        for key in vars(grouped_instance).keys():
            if key not in _SERDE_EXCLUDE.union(_SERDE_PICKLE_OBJ):
                model_encode[key] = str(getattr(grouped_instance, key))
        for attr in _SERDE_PICKLE_OBJ:
            model_encode[attr] = self._encode_obj(getattr(grouped_instance, attr))
        return model_encode


class ModelDecoder:
    def __init__(self, model_json):
        self.model_json = model_json

    @staticmethod
    def _decode_obj(obj):

        return {
            literal_eval(key): pickle.loads(literal_eval(value))
            for key, value in obj.items()
        }

    def decode(self):

        model_decode = {}
        payload_eval = self.model_json
        for key, value in payload_eval.items():
            if key not in _SERDE_EXCLUDE.union(_SERDE_PICKLE_OBJ):
                try:  # literal_eval can't evaluate values that are already primitives
                    model_decode[key] = literal_eval(value)
                except ValueError:
                    model_decode[key] = value
            if key in _SERDE_PICKLE_OBJ:
                model_decode[key] = self._decode_obj(value)
        model_decode["model_clazz"] = _get_statsmodels_model(payload_eval["model_type"])
        return model_decode


def group_statsmodels_save(model, path):

    encoded = ModelEncoder().encode(model)

    with open(path, "w") as f:
        json.dump(encoded, f)


def group_statsmodels_load(path):

    with open(path, "r") as f:
        model = json.load(f)
    decoded_dict = ModelDecoder(model).decode()

    return decoded_dict
