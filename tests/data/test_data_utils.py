from tests import data_generator
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.data.utils.dataframe_utils import apply_datetime_index_to_groups
import pandas as pd


def test_datetime_indexing():

    data = data_generator.generate_test_data(3, 4, 1000, "2020-01-01", 6)

    grouped_data = PandasGroupGenerator(data.key_columns).generate_processing_groups(
        data.df
    )

    indexed_data = apply_datetime_index_to_groups(grouped_data, "ds")

    for key, df in indexed_data:
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.freqstr == "6D"
