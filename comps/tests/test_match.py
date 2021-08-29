import pytest

import numpy as np
import pandas as pd

from mystique.match import array_matches


def test_array_matches(int_array):
    """Test matching of arrays"""
    array1 = int_array[0]
    array2 = np.zeros(4)
    array3 = np.array([1, 3, 5, 7])
    assert np.array_equal(array_matches(array1, int_array), np.array([0, 5, 9]))
    assert np.array_equal(array_matches(array2, int_array), np.empty(0))
    assert np.array_equal(array_matches(array3, int_array), np.array([2, 7]))

def test_greedy_match(int_array):
    """Test ranking of arrays"""
    pass
