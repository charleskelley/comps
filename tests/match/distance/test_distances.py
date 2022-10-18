import pytest

from numpy import ndarray, reshape, where
from sklearn.linear_model import LogisticRegression

from comps.match.distance.distances import Distances


pystestmark = pytest.mark.unit


@pytest.fixture
def lalonde_skdistances(lalonde_pandas, lalonde_columns):
    """Default scikit-learn logistic regression model fit to Lalonde data"""
    features = lalonde_pandas[lalonde_columns["features"]].to_numpy(float)
    targets = lalonde_pandas[lalonde_columns["target"]].to_numpy(float)

    rows = {"target": where(targets == 1)[0], "non_target": where(targets == 0)[0]}
    uids = {
        "name": "observation",
        "target": rows["target"] + 1,
        "non_target": rows["non_target"] + 1,
    }

    model = LogisticRegression().fit(features, targets)

    target_probabilities = model.predict_proba(features[targets == 1])[:, 1]
    non_target_probabilities = model.predict_proba(features[targets == 0])[:, 1]
    propensities = {
        "target": target_probabilities,
        "non_target": non_target_probabilities,
    }

    target_propensities = reshape(
        target_probabilities,
        (len(target_probabilities), 1),
    )
    non_target_propensities = reshape(
        non_target_probabilities,
        (1, len(non_target_probabilities)),
    )
    distances = abs(target_propensities - non_target_propensities)  # type: ignore

    return Distances(
        "logistic",
        distances,
        rows,  # type: ignore
        lalonde_columns["target"],
        lalonde_columns["features"],
        model,
        propensities,
        uids,  # type: ignore
    )


def test_skdistances_init(lalonde_skdistances):
    """Test initialization of Distances class using sklearn/numpy attributes"""
    assert isinstance(lalonde_skdistances, Distances)


def test_data_row_pairs(lalonde_skdistances):
    """Normalization of row numbers into cartesian cross of target x non_target"""
    data_row_pairs = lalonde_skdistances._data_row_pairs()
    assert isinstance(data_row_pairs, ndarray)
    assert len(data_row_pairs) == 297 * 425


def test_rows_index_pairs(lalonde_skdistances):
    """
    Normalization of row counts into cartesian cross of target x non_target
    index position in target x non-target pairwise m x n 2D ndarray.
    """
    rows_index_pairs = lalonde_skdistances._rows_index_pairs()
    assert isinstance(rows_index_pairs, ndarray)
    assert len(rows_index_pairs) == 297 * 425
    assert rows_index_pairs.shape == (297 * 425, 2)
    assert max(rows_index_pairs[:, 0]) + 1 == 297
    assert max(rows_index_pairs[:, 1]) + 1 == 425


def test_uid_pairs(lalonde_skdistances):
    """Normalization of uids into cartesian cross of target x non_target"""
    uid_pairs = lalonde_skdistances._uid_pairs()
    assert isinstance(uid_pairs, ndarray)
    assert len(uid_pairs) == 297 * 425


def test_pair_distance(lalonde_skdistances):
    """
    Normalize pairwise row x column pairwise distance to target x non-target
    combination distance as single column.
    """
    rows_index_pairs = lalonde_skdistances._rows_index_pairs()
    pair_distance = lalonde_skdistances._pair_distance(rows_index_pairs)

    assert len(pair_distance) == len(rows_index_pairs)
    assert 0 <= pair_distance.all() <= 1


def test_pair_propensities(lalonde_skdistances):
    """Target and non-target pairs observation propensity scores"""
    rows_indices = lalonde_skdistances._rows_index_pairs()
    pair_propensities = lalonde_skdistances._pair_propensities(rows_indices)

    assert len(rows_indices) == len(rows_indices)
    assert 0 <= pair_propensities.all() <= 1


def test_distances_pandas(lalonde_skdistances):
    """Pandas DataFrame summary of data from the distances calculation scenario"""
    dataframe = lalonde_skdistances._distances_pandas()

    assert list(dataframe.columns) == [
        "target_observation",
        "non_target_observation",
        "target_pair",
        "distance",
        "target_propensity",
        "non_target_propensity",
    ]

    assert len(dataframe) == 297 * 425

    assert dataframe.target_observation.min() == 1
    assert dataframe.target_observation.max() == 297
    assert dataframe.non_target_observation.min() == 298
    assert dataframe.non_target_observation.max() == 722
