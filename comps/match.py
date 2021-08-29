"""
Algorithms for for matching on single and multiple variables 
"""
import numpy as np

# Data for testing
from sklearn import datasets
diabetes = datasets.load_diabetes()

from sklearn.utils.validation import check_array, check_X_y
from sklearn.neighbors import NearestNeighbors
# from scipy.stats import rankdata

# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import maximum_bipartite_matching
# scipy.sparse.csgraph import min_weight_full_bipartite_matching
# from scipy.optimize import linear_sum_assignment


def array_matches(a, X, first_only=False, equal_nan=False):
    """Returns index of matches of 1D array in another 2D array of arrays. 

    The 2D array of arrays must be the same length as the 1D array so a proper
    comparison can be made.  

    Parameters
    ----------
    a : array
        Target array to find matches for.

    X : 2D array
        Array of arrays to search for matches array matches in.

    first_only : bool
        Returns only the index of first array match encountered if True.
 
    equal_nan : bool
        Whether to compare NaN’s as equal.

    Returns
    -------
    matches : array
        NumPy array of integers with index numbers of all matching arrays in X. 

    int
        Integer index of first match in array of arrays X.
    """
    X = check_array(X)
    a = check_array(a, ensure_2d=False)
    assert X.shape[1] == a.shape[0], ("Comparison '1D' array a must be same "
        "length as each array in 2D array 'X'")
    if a.ndim != 1:
        ValueError("{0} must be 1D array to compare to 2D array of arrays X".format(a))

    matches = np.zeros(len(X), dtype="int64")
    match_count = 0

    for idx, arr in enumerate(X):
        if np.array_equal(a, arr, equal_nan=equal_nan):
            matches[match_count] = idx
            match_count += 1
        if first_only and match_count == 1:
            matches = matches[0]
            return matches
    matches = np.trim_zeros(matches, trim="b")
    return matches


class GreedyMatch:
    """Class for greedy or nearest neighbors matching.

    This class extends the sklearn.neighbors.NearestNeighbors class and can be
    initialized with the NearestNeighbors class parameters.

    Parameters
    ----------
    X : {ndarray, list, sparse matrix}
        Input or training data to match on.

    uid : {ndarray, list, sparse matrix}
        Unique ID labels for X to report matching for.
    
    *args : varies
        Positional arguments to sklearn.neighbors.NearestNeighbors.

    **kwargs : varies
        Keyword arguments to sklearn.neighbors.NearestNeighbors.

    Attributes
    ----------
    matcher : mystique.xfmr.Xfmr 
        Transformer class that tracks data for  
    """
    def __init__(self, X=None, uid=None, *args, **kwargs):

        self.neigh = NearestNeighbors(*args, **kwargs)

        if isinstance(X, np.ndarray):
            if uid:
                check_X_y(X, uid, multi_output=True)
            else:
                check_array(X)
            self.neigh.fit(X)

        self.uid = uid
        self.matcher = Xfmr()

    def greedy_match(self, X=None, targets=None, controls=None, **kwargs):
        """Greedy matching using sklearn.neighbors.NearestNeighbors.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), or (n_queries, n_indexed)
        if metric == ‘precomputed’, default=None 
            The query point or points. If not provided, neighbors of each
            indexed point are returned. In this case, the query point is not
            considered its own neighbor.

        targets : array-like
            Arrray of boolean resolvable indicators of which rows in X are
            target observations to be matched. Non true observations are
            assumed to be control observations
 
        controls : array-like
            Arrray of boolean resolvable indicators of which rows in X are
            control observations to be matched. Non true observations are
            assumed to be target observations.

        exact : bool or dict
            Whether to use exact matching.

        replace : bool
            Whether to match with replacement. Default is `False`. If `True`,
            the indexed order 0 to N of the query points in X will dictate the
            order of matches removed from the eligiible matching points set.

        selection : str
            Type of selection method when selecting k:1 without replacement.
            Can be <'lowest'|'roundrobin'>
            

        caliper : float
            Range of parameter space to use by default for radius_neighbors queries.

        k : int
            Number of matches to retrieve for target group.

        distance : str
            Type of distance metric to use. See sklearn.neighbors.DistanceMetric
            documentation for a list of valid distance metrics.

        Returns
        -------
        matches : ndarray of shape (n_queries, n_neighbors)
            
        """
        if np.any(X):
            self.neigh.fit(X)

        return self.neigh.kneighbors_graph(X, mode="distance", **kwargs)


