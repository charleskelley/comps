import pytest
from sklearn.linear_model import LogisticRegression

from comps.match.distance.engine import Engine
from comps.match.distance.sklearn import SklearnDistance


pystestmark = pytest.mark.unit


@pytest.fixture
def skdistance():
    """Initialized instance of SklearnDistance class Engine"""
    sklearn_engine = SklearnDistance()

    return sklearn_engine


@pytest.fixture
def lalonde_pandas(data_factory):
    """Pandas DataFrame with Lalonde NSW data"""
    return data_factory("lalonde", "pandas")


@pytest.fixture
def lalonde_columns():
    """Key sets of columns from Lalonde data"""
    columns = {
        "features": [
            "age",
            "education",
            "black",
            "hispanic",
            "married",
            "nodegree",
            "re75",
        ],
        "target": "treatment",
        "uid": "observation",
    }

    return columns


@pytest.fixture
def lalonde_logistic_model(lalonde_pandas, lalonde_columns):
    """Default scikit-learn Logistic regression model fit to Lalonde data"""
    features = lalonde_pandas[lalonde_columns["features"]].to_numpy(float)
    targets = lalonde_pandas[lalonde_columns["target"]].to_numpy(float)

    return LogisticRegression().fit(features, targets)


def test_sklearn_distance_init():
    """Class initialization and abstract base class inheritance"""
    sklearn_engine = SklearnDistance()
    assert isinstance(sklearn_engine, SklearnDistance)
    assert isinstance(sklearn_engine, Engine)


def test_class_rows(skdistance, lalonde_pandas):
    """Identifying target and non-target observation row numbers"""
    rows = skdistance._class_rows(lalonde_pandas, "treatment")

    assert len(rows["target"]) == 297
    assert (min(rows["target"]), max(rows["target"])) == (0, 296)

    assert len(rows["non_target"]) == 425
    assert (min(rows["non_target"]), max(rows["non_target"])) == (297, 721)


def test_class_uids(skdistance, lalonde_pandas):
    """Identifying target and non-target observation row numbers"""
    uids = skdistance._class_uids(lalonde_pandas, "treatment", "observation")

    assert uids["name"] == "observation"

    assert len(uids["target"]) == 297
    assert (min(uids["target"]), max(uids["target"])) == (1, 297)

    assert len(uids["non_target"]) == 425
    assert (min(uids["non_target"]), max(uids["non_target"])) == (298, 722)


def test_class_propensities(
    skdistance, lalonde_pandas, lalonde_columns, lalonde_logistic_model
):
    """Propensity score arrays for target and non-target observations"""
    propensities = skdistance._class_propensities(
        lalonde_logistic_model,
        lalonde_pandas,
        lalonde_columns["target"],
        lalonde_columns["features"],
    )

    target_count = len(lalonde_pandas[lalonde_pandas[lalonde_columns["target"]] == 1])
    non_target_count = len(
        lalonde_pandas[lalonde_pandas[lalonde_columns["target"]] == 0]
    )

    assert (
        len(propensities["target"]) + len(propensities["non_target"])
        == lalonde_pandas.shape[0]
    )
    assert len(propensities["target"]) == target_count
    assert len(propensities["non_target"]) == non_target_count

    assert propensities["target"].all() >= 0 and propensities["target"].all() <= 1
    assert (
        propensities["non_target"].all() >= 0 and propensities["non_target"].all() <= 1
    )


def test_propensity_distances(
    skdistance, lalonde_pandas, lalonde_columns, lalonde_logistic_model
):
    """Pairwise m x n (target x non_target) differences between propensities"""
    propensities = skdistance._class_propensities(
        lalonde_logistic_model,
        lalonde_pandas,
        lalonde_columns["target"],
        lalonde_columns["features"],
    )

    target_count = len(lalonde_pandas[lalonde_pandas[lalonde_columns["target"]] == 1])
    non_target_count = len(
        lalonde_pandas[lalonde_pandas[lalonde_columns["target"]] == 0]
    )

    distances = skdistance._propensity_distances(propensities)

    assert distances.shape == (target_count, non_target_count)
    assert distances.all() >= 0 and distances.all() <= 1


def test_features(skdistance, lalonde_pandas, lalonde_columns):
    """Default identification of undeclared column names from DataFrame"""
    with pytest.raises(TypeError):
        skdistance._features(lalonde_pandas, lalonde_columns["features"])

    features_passed_through = skdistance._features(
        lalonde_pandas, lalonde_columns["target"], lalonde_columns["features"]
    )

    assert features_passed_through == lalonde_columns["features"]

    expected_default_features = ["dataset"] + lalonde_columns["features"] + ["re78"]
    default_features = skdistance._features(
        lalonde_pandas,
        lalonde_columns["target"],
        uid=lalonde_columns["uid"],
    )
    assert default_features == expected_default_features


def test_fit(skdistance, lalonde_pandas, lalonde_columns):
    """Fitting algorithms and storing models in models attribute"""

    # Default hyperparameters
    skdistance.fit(
        "logistic",
        lalonde_pandas,
        lalonde_columns["target"],
        lalonde_columns["features"],
    )

    assert isinstance(skdistance.models["logistic"], LogisticRegression)
    assert skdistance.models["logistic"].coef_.all()
    assert len(skdistance.models["logistic"].coef_[0]) == 7

    # Pass hyperparameters to algorithm class


# def test_propensity_score_model(data_factory, lalonde_variables):
#     lalonde_pandas = data_factory("lalonde", "pandas")
#     feature_names = lalonde_variables["features"]

#     # Check with ndarray
#     features = lalonde_pandas[feature_names].to_numpy(float)
#     targets = lalonde_pandas[lalonde_variables["target"]].to_numpy(float)
#     psm_estimator = method.propensity_score_model(features, targets)
#     assert isinstance(psm_estimator, LogisticRegression)
#     assert psm_estimator.coef_.all()

#     # Check with Pandas
#     features = lalonde_pandas[feature_names].astype(float)
#     targets = lalonde_pandas[lalonde_variables["target"]].astype(float)
#     psm_estimator = method.propensity_score_model(features, targets)
#     assert isinstance(psm_estimator, LogisticRegression)
#     assert psm_estimator.coef_.all()
#     assert psm_estimator.feature_names_in_ == feature_names


# def test_pairwise_distance(data_factory):
#     pass
