"""
Test module for evaluating functionality within `diviner.utils.common`
"""
import pytest
import pandas as pd
from diviner.utils.common import _validate_keys_in_df
from diviner.exceptions import DivinerException


def generate_sample():
    """
    Utility function to generate a simple example Pandas DataFrame for key validation purposes.
    :return: pd.DataFrame
    """
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
    """
    Test for ensuring that grouping keys are properly validated and that invalid keys submitted
    for validation will raise a DivinerException with the appropriate message.
    :return: None
    """
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
        _validate_keys_in_df(df, invalid_group_keys)

    assert _validate_keys_in_df(df, valid_subset_group_keys) is None
    assert _validate_keys_in_df(df, valid_group_keys) is None
