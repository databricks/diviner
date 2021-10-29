import abc
from typing import List


class BaseGroupGenerator(abc.ABC):
    def __init__(self, group_key_columns: List[str]):
        self.group_key_columns = group_key_columns
        self.master_group_key = "grouping_key"

    @abc.abstractmethod
    def generate_processing_groups(self, df):
        pass
