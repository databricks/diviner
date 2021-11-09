import pandas as pd
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.exceptions import DivinerException
import pytest


def generate_sample():
    data = {
        "ds": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "y": [12.0, 13.0, 14.0],
        "a": [1, 1, 2],
        "b": [2, 2, 2],
    }

    return pd.DataFrame.from_dict(data)


def test_pandas_group_generator_master_key_creation():
    data = generate_sample()
    grouping_keys = ("a", "b")
    master_key_add = PandasGroupGenerator(grouping_keys)._create_master_key_column(data)

    for i in range(len(master_key_add)):
        row = master_key_add.iloc[i]
        constructed_master = tuple([row[x] for x in grouping_keys])
        assert constructed_master == row["grouping_key"]


def test_pandas_group_data_creation():
    data = generate_sample()
    grouping_keys = ("a", "b")

    group_gen = PandasGroupGenerator(grouping_keys).generate_processing_groups(data)

    updated = data.copy()
    # disable linting because of a pandas tuple implementation bug:
    #   https://github.com/PyCQA/pylint/issues/1709
    updated["grouping_key"] = updated[[*grouping_keys]].apply(
        lambda x: tuple(x), axis=1  # pylint: disable=unnecessary-lambda
    )
    raw_tuple_keys = set(updated["grouping_key"])

    assert len(group_gen) == 2
    assert set([x for x, v in group_gen]).issubset(raw_tuple_keys)


def test_pandas_group_data_generator_invalid_group_keys():

    data = generate_sample()
    grouping_keys = ()

    with pytest.raises(
        DivinerException,
        match="Argument 'group_key_columns' tuple must contain at "
        "least one string entry.",
    ):
        PandasGroupGenerator(grouping_keys).generate_processing_groups(data)
