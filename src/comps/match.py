"""
Algorithms for for matching on single and multiple variables
"""
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array


# from scipy.stats import rankdata

# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import maximum_bipartite_matching
# scipy.sparse.csgraph import min_weight_full_bipartite_matching
# from scipy.optimize import linear_sum_assignment


def array_matches(
    a: NDArray[Any], X: NDArray[Any], first_only: bool = False, equal_nan: bool = False
) -> NDArray[Any]:
    """
    Returns index of matches of 1D array in another 2D array of arrays. The 2D
    array of arrays must be the same length as the 1D array so a proper
    comparison can be made.

    Args:
        a: Target array to find matches for.
        X: 2D array of arrays to search for matches array matches in.
        first_only: Returns only the index of first array match encountered if True.
        equal_nan: Whether to compare NaN’s as equal.

    Returns:
        matches: Array of integers with index numbers of all matching arrays in X.
    """
    X = check_array(X)
    a = check_array(a, ensure_2d=False)

    if not X.shape[1] == a.shape[0]:
        raise AssertionError(
            "Comparison '1D' array a must be same length as each array in 2D array 'X'"
        )

    if a.ndim != 1:
        ValueError("{0} must be 1D array to compare to 2D array of arrays X".format(a))

    matches = np.zeros(len(X), dtype="int64")  # type: NDArray[Any]
    match_count = 0

    for idx, arr in enumerate(X):

        if np.array_equal(a, arr, equal_nan=equal_nan):
            matches[match_count] = idx
            match_count += 1

        if first_only and match_count == 1:
            matches = matches[0]
            return matches

    return np.trim_zeros(matches, trim="b")
