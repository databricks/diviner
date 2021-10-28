from typing import List
from diviner.exceptions import DivinerException


def validate_group_key_schema(schema: List[str], group_keys: List[str]) -> None:

    if not all(column in schema for column in group_keys):
        raise DivinerException(
            f"Schema provided: '{schema} does not contain provided "
            f"grouping key columns: '{group_keys}'"
        )
