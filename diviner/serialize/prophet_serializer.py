"""
Module for serialization and deserialization for GroupedProphet models.
This implementation uses JSON strings for serialization to aid in user legibility of the
saved model.
"""
import os
import json
from ast import literal_eval
from prophet.serialize import model_from_json, model_to_json
from diviner.exceptions import DivinerException

GROUPED_MODEL_ATTRIBUTES = ["group_key_columns", "master_key"]


def _grouped_model_to_dict(grouped_model):

    model_dict = {
        attr: getattr(grouped_model, attr) for attr in GROUPED_MODEL_ATTRIBUTES
    }
    model_dict["model"] = {
        str(master_key): model_to_json(model)
        for master_key, model in grouped_model.model.items()
    }
    return model_dict


def grouped_model_to_json(grouped_model):
    """
    Serialization helper function to convert a GroupedProphet instance to json for saving to disk.

    :param grouped_model: Instance of GroupedProphet() that has been fit.
    :return: serialized json string of the model's attributes
    """

    model_dict = _grouped_model_to_dict(grouped_model)
    for key in vars(grouped_model).keys():
        if key != "model":
            model_dict[key] = getattr(grouped_model, key)

    return json.dumps(model_dict)


def _grouped_model_from_dict(raw_model):

    deser_model_payload = {
        literal_eval(master_key): model_from_json(payload)
        for master_key, payload in raw_model.items()
    }
    return deser_model_payload


def grouped_model_from_json(path):
    """
    Helper function to load the grouped model structure from serialized json and deserialize
    the Prophet instances.

    :param path: The storage location of a saved GroupedProphet object
    :return: Dictionary of instance attributes
    """
    if not os.path.isfile(path):
        raise DivinerException(
            f"There is no valid model saved at the specified path: {path}"
        )
    with open(path, "r") as f:
        raw_model = json.load(f)

    model_deser = _grouped_model_from_dict(raw_model["model"])
    raw_model["model"] = model_deser

    return raw_model
