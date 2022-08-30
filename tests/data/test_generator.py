import pandas as pd
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.exceptions import DivinerException
import pytest


@pytest.fixture(scope="module")
def data():
    data = {
        "ds": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-01"],
        "y": [12.0, 13.0, 14.0, 35.0],
        "a": [1, 1, 1, 2],
        "b": [1, 1, 2, 2],
    }

    return pd.DataFrame.from_dict(data)


def test_pandas_group_generator_master_key_creation(data):
    grouping_keys = ("a", "b")
    master_key_add = PandasGroupGenerator(grouping_keys, "ds", "y")._get_df_with_master_key_column(
        data
    )

    for i in range(len(master_key_add)):
        row = master_key_add.iloc[i]
        constructed_master = tuple([row[x] for x in grouping_keys])
        assert constructed_master == row["grouping_key"]


def test_pandas_group_data_creation(data):
    grouping_keys = ("a", "b")

    group_gen = PandasGroupGenerator(grouping_keys, "ds", "y").generate_processing_groups(data)

    updated = data.copy()
    # disable linting because of a pandas tuple implementation bug:
    #   https://github.com/PyCQA/pylint/issues/1709
    updated["grouping_key"] = updated[[*grouping_keys]].apply(
        lambda x: tuple(x), axis=1  # pylint: disable=unnecessary-lambda
    )
    raw_tuple_keys = set(updated["grouping_key"])

    assert len(group_gen) == 3
    assert set([x for x, v in group_gen]).issubset(raw_tuple_keys)


def test_pandas_group_data_generator_invalid_group_keys(data):

    grouping_keys = ()

    with pytest.raises(
        DivinerException,
        match="Argument '_group_key_columns' tuple must contain at " "least one string entry.",
    ):
        PandasGroupGenerator(grouping_keys, "ds", "y").generate_processing_groups(data)


def test_pandas_group_data_generator_partial_keys(data):
    grouping_keys = ("a",)

    group_gen = PandasGroupGenerator(
        grouping_keys, datetime_col="ds", y_col="y"
    ).generate_processing_groups(data)

    assert len(group_gen) == 2
    first_group, first_df = group_gen[0]
    assert first_group == (1,)
    assert len(first_df) == 2
    assert first_df["y"].sum() == 39
    second_group, second_df = group_gen[1]
    assert second_group == (2,)
    assert len(second_df) == 1
    assert second_df["y"].sum() == 35
