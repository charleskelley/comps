import os
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

@pytest.fixture
def pd_bank_data():
    """Returns bank data in pandas DataFrame for testing.

    Returns
    -------
    pandas : DataFrame 
        Pandas DataFrame with bank data.
    """
    return pd.read_csv(
        Path(__file__).resolve.parent.joinpath("datasets", "bank", "bank.csv"),
        sep=";", header=0).convert_dtypes()
        

@pytest.fixture
def int_array():
    """Returns 10x4 NumPy array for testing.

    Returns
    -------
    numpy : Array 
        10x4 array with int 0 to 9 for each value.
    """
    return np.array([
        [0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 5, 7], [7, 5, 3, 1], [2, 4, 6, 8],
        [0, 1, 2, 3], [8, 6, 4, 2], [1, 3, 5, 7], [3, 5, 7, 9], [0, 1, 2, 3], ],
        np.int64)

