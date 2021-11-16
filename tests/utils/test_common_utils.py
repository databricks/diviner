from diviner.utils.common import validate_keys_in_df
import pandas as pd
from diviner.exceptions import DivinerException
import pytest


def generate_sample():
    data = {
        "ds": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "y": [12.0, 13.0, 14.0],
        "a": [1, 1, 2],
        "b": [2, 2, 2],
        "c": [1, 1, 1],
        "z": [37, 42, 42],
    }

    return pd.DataFrame.from_dict(data)


def test_validate_keys_in_df():

    invalid_group_keys = ("a", "b", "q")
    valid_subset_group_keys = (
        "a",
        "b",
        "c",
    )  # missing the 'z' column but we don't raise on that.
    valid_group_keys = ("a", "b", "c", "z")

    df = generate_sample()

    with pytest.raises(
        DivinerException,
        match=(
            "Not all key grouping columns supplied: \\('a', 'b', 'q'\\) are present "
            "in the submitted df: \\['ds', 'y', 'a', 'b', 'c', 'z'\\]"
        ),
    ):
        validate_keys_in_df(df, invalid_group_keys)

    assert validate_keys_in_df(df, valid_subset_group_keys) is None
    assert validate_keys_in_df(df, valid_group_keys) is None

def test_predict_conf_df_validation():

