"""
Serialization and Deserialization tests for GroupedProphet model type
"""
from copy import deepcopy
import os
from diviner import GroupedProphet
from tests import data_generator


def test_prophet_save_load_override_object():
    """Test to ensure that deserialization updates object properly for all attributes"""

    train1 = data_generator.generate_test_data(3, 2, 1000, "2020-01-01", 1)
    train2 = data_generator.generate_test_data(2, 2, 500, "2021-01-01", 1)

    model1 = GroupedProphet().fit(train1.df, train1.key_columns)
    model2 = GroupedProphet().fit(train2.df, train2.key_columns)

    model1_group_keys = deepcopy(model1._group_key_columns)
    model1_model = deepcopy(model1.model)

    # save model 2
    save_path = os.path.join("/tmp/group_prophet_test", "model2serdetest.gpm")
    model2.save(save_path)

    # use model1 object to load model2
    reloaded = model1.load(save_path)

    assert set(reloaded._group_key_columns) != set(model1_group_keys)
    assert reloaded.model.keys() == model2.model.keys()
    assert reloaded.model.keys() != model1_model.keys()
