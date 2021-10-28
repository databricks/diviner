import json
import importlib
from prophet.serialize import model_from_json, model_to_json
from ast import literal_eval

GROUPED_MODEL_ATTRIBUTES = ["group_key_columns", "master_key"]


def _grouped_model_serialize(grouped_model):

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

    model_dict = _grouped_model_serialize(grouped_model)
    return json.dumps(model_dict)


def _grouped_model_deserialize(model_dict, module, clazz):

    init_attr = {key: model_dict[key] for key in GROUPED_MODEL_ATTRIBUTES}
    raw_model_payload = model_dict["model"]
    deser_model_payload = {
        literal_eval(master_key): model_from_json(payload)
        for master_key, payload in raw_model_payload.items()
    }

    module_ = importlib.import_module(module)
    model = getattr(module_, clazz)()
    setattr(model, "master_key", init_attr["master_key"])
    setattr(model, "group_key_columns", init_attr["group_key_columns"])
    setattr(model, "model", deser_model_payload)

    return model


def grouped_model_from_json(model_json, module, clazz):
    """
    Helper function to load the grouped model structure from serialized json as an instance
    of GroupedProphet()
    :param model_json: The json string serialized instance of a saved GroupedProphet model
    :param module: The package module name for the model type
    :param clazz: The class instance of the model
    :return: Instance of GroupedProphet that applies the class arguments from the saved model
    """

    model_dict = json.loads(model_json)

    return _grouped_model_deserialize(model_dict, module, clazz)
