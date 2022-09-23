"""
Different types of data summaries to use for exploratory analysis
"""
from typing import Any, Optional, Sequence, Union

from pandas import DataFrame


# Five-number min, 25th, median, 75th, max

# Frequency
# variable name | freq, pct, cum_freq, cum_pct
# crosstable | freq, pct, row_pct, col_pct


def frequency(
    data: Any, variables: Union[str, Sequence[str]], return_data: bool = False
) -> Optional[DataFrame]:
    pass


def distribution(
    data: Any, variables: Union[str, Sequence[str]]
) -> Optional[DataFrame]:
    pass
