from diviner.data.base_generator import BaseGroupGenerator
from diviner.data.utils import validate_group_key_schema
import pandas as pd
from typing import List


class PandasGroupGenerator(BaseGroupGenerator):
    def __init__(self, group_key_columns: List[str]):
        super().__init__(group_key_columns)

    def _create_master_key_column(self, df) -> pd.DataFrame:

        validate_group_key_schema(
            schema=list(df.columns), group_keys=self.group_key_columns
        )

        master_group_df = df.copy()
        master_group_df[self.master_group_key] = master_group_df[
            [*self.group_key_columns]
        ].apply(lambda column: tuple(column), axis=1)
        return master_group_df

    def generate_processing_groups(self, df):

        master_key_generation = self._create_master_key_column(df)

        grouped_data = list(
            dict(tuple(master_key_generation.groupby(self.master_group_key))).items()
        )

        return grouped_data
