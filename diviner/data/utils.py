from typing import List
from diviner.exceptions import DivinerException


def validate_group_key_schema(schema: List[str], group_keys: List[str]) -> None:

    if not set(group_keys).issubset(set(schema)):
        raise DivinerException(
            f"Schema provided: {schema} does not contain provided "
            f"grouping key columns: {group_keys}"
        )
