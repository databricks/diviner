import pickle
import json
from ast import literal_eval

_PMDARIMA_CLASS_DICTS = {"model", "_max_datetime_per_group"}
_PMDARIMA_CLASS_OBJECTS = {"_model_constructor"}


class PmdarimaEncoder:
    def __init__(self):
        self.serialized_model = {}

    def _encode_dict(self, name, obj):
        self.serialized_model[name] = {
            str(key): str(pickle.dumps(value)) for key, value in obj.items()
        }

    def _encode_obj(self, name, obj):
        self.serialized_model[name] = str(pickle.dumps(obj))

    def encode(self, grouped_model):
        for key in vars(grouped_model).keys():
            value = getattr(grouped_model, key)
            if value:
                if key not in _PMDARIMA_CLASS_DICTS.union(_PMDARIMA_CLASS_OBJECTS):
                    self.serialized_model[key] = str(value)
                if key in _PMDARIMA_CLASS_DICTS:
                    self._encode_dict(key, value)
                if key in _PMDARIMA_CLASS_OBJECTS:
                    self._encode_obj(key, value)
            else:
                self.serialized_model[key] = None
        return self.serialized_model


class PmdarimaDecoder:
    def __init__(self):
        self.model_decode = {}

    def _decode_obj(self, key, obj):
        self.model_decode[key] = pickle.loads(literal_eval(obj))

    def _decode_dict(self, key, obj):
        self.model_decode[key] = {
            literal_eval(key): pickle.loads(literal_eval(value))
            for key, value in obj.items()
        }

    def decode(self, model_json):
        for key, value in model_json.items():
            if value:
                if key not in _PMDARIMA_CLASS_DICTS.union(_PMDARIMA_CLASS_OBJECTS):
                    try:
                        self.model_decode[key] = literal_eval(value)
                    except ValueError:
                        self.model_decode[key] = value
                if key in _PMDARIMA_CLASS_DICTS:
                    self._decode_dict(key, value)
                if key in _PMDARIMA_CLASS_OBJECTS:
                    self._decode_obj(key, value)
            else:
                self.model_decode[key] = None
        return self.model_decode


def group_pmdarima_save(model, path):

    encoded = PmdarimaEncoder().encode(model)

    with open(path, "w") as f:
        json.dump(encoded, f)


def group_pmdarima_load(path):

    with open(path, "r") as f:
        model = json.load(f)

    decoded = PmdarimaDecoder().decode(model)
    return decoded
