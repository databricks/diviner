import pickle
import json
from base64 import urlsafe_b64decode, urlsafe_b64encode

_PMDARIMA_CLASS_DICTS = {
    "model",
    "_max_datetime_per_group",
    "_datetime_freq_per_group",
    "_ndiffs",
    "_nsdiffs",
}
_PMDARIMA_CLASS_OBJECTS = {"_model_template"}


class PmdarimaEncoder:
    def __init__(self):
        self.serialized_model = {}

    @staticmethod
    def _byte_encode(payload):
        return urlsafe_b64encode(pickle.dumps(payload)).decode("utf-8")

    def _encode_dict(self, name, obj):
        self.serialized_model[name] = {
            self._byte_encode(key): self._byte_encode(value) for key, value in obj.items()
        }

    def _encode_obj(self, name, obj):
        self.serialized_model[name] = self._byte_encode(obj)

    def encode(self, grouped_model):
        for key in vars(grouped_model).keys():
            value = getattr(grouped_model, key)
            if value:
                if key not in _PMDARIMA_CLASS_DICTS.union(_PMDARIMA_CLASS_OBJECTS):
                    self.serialized_model[key] = value
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

    @staticmethod
    def _byte_decode(payload):
        return pickle.loads(urlsafe_b64decode(payload.encode("utf-8")))

    def _decode_obj(self, key, obj):
        self.model_decode[key] = self._byte_decode(obj)

    def _decode_dict(self, key, obj):
        self.model_decode[key] = {
            self._byte_decode(key): self._byte_decode(value) for key, value in obj.items()
        }

    def decode(self, model_json):
        for key, value in model_json.items():
            if value:
                if key not in _PMDARIMA_CLASS_DICTS.union(_PMDARIMA_CLASS_OBJECTS):
                    self.model_decode[key] = value
                if key in _PMDARIMA_CLASS_DICTS:
                    self._decode_dict(key, value)
                if key in _PMDARIMA_CLASS_OBJECTS:
                    self._decode_obj(key, value)
            else:
                self.model_decode[key] = None
        return self.model_decode


def grouped_pmdarima_save(model, path):

    encoded = PmdarimaEncoder().encode(model)

    with open(path, "w") as f:
        json.dump(encoded, f)


def grouped_pmdarima_load(path):

    with open(path, "r") as f:
        model = json.load(f)

    decoded = PmdarimaDecoder().decode(model)
    return decoded
