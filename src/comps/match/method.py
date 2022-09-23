"""
Module of different matching methods used to match observations based on
feature similarity.
"""
from numpy import ndarray
from sklearn.metrics import pairwise_distances


def stratified_matches():
    pass


def propenstity_cost_matrix():
    pass


def distance_cost_matrix(
    test_targets: ndarray,
    control_eligible: ndarray,
    distance_metric: str = "mahalanobis",
    **kwds,
) -> ndarray:
    """

    Args:
        test_targets: Numeric feature matrix for all test group observations
            that should be targeted for matches to the control eligible
            observations

        control_eligible: Numeric feature matrix for all control group eligible
            observations.

        distance_metric:

        **kwds: Optional keyword parameters to ``sklearn.pairwise_distances``
            that will be passed directly to the distance function. If using a
            scipy.spatial.distance metric, the parameters are still metric
            dependent. See the scipy docs for usage examples.

    Returns:
        An ndarray of shape (n_test_eligible, n_test_eligible) or
            (n_test_eligible, n_control_eligible) A distance matrix D such that
            D_{i, j} is the distance between the ith and jth vectors of the
            given matrix X, if Y is None. If Y is not None, then D_{i, j} is
            the distance between the ith array from X and the jth array from Y.
    """
    return pairwise_distances(
        test_targets,
        control_eligible,
        metric=distance_metric,
        n_jobs=-1,
        **kwds,
    )
