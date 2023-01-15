"""
Tests for the ``comps.match.distance.sklearn`` module.
"""
from importlib import import_module

import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from comps.match.distance.sklearn import (SKLEARN_DISTANCE_ALGORITHMS,
                                          SklearnDistance)

pystestmark = pytest.mark.unit


@pytest.fixture
def sklearn_distance():
    """Initialized instance of SklearnDistance class Engine"""
    sklearn_engine = SklearnDistance()

    return sklearn_engine


@pytest.fixture
def lalonde_logistic_model(lalonde_pandas, lalonde_columns):
    """Default scikit-learn Logistic regression model fit to Lalonde data"""
    features = lalonde_pandas[lalonde_columns["features"]].to_numpy(float)
    targets = lalonde_pandas[lalonde_columns["target"]].to_numpy(int)

    return LogisticRegression().fit(features, targets)


@pytest.fixture
def lalonde_data(lalonde_pandas, lalonde_columns):
    """Input DataFrame for all SklearnDistance methods"""
    input_data = lalonde_pandas[
        [lalonde_columns["target"]] + lalonde_columns["features"]
    ].astype(float)

    return input_data


def test_sklearn_distance_init():
    """Class initialization and abstract base class inheritance"""
    sklearn_engine = SklearnDistance()
    assert isinstance(sklearn_engine, SklearnDistance)


def test_model(sklearn_distance, lalonde_data):
    """Fitting algorithms and storing models in models attribute"""
    # Default algorithm class hyperparameters
    model = sklearn_distance.model(
        lalonde_data,
        "logistic",
    )

    assert isinstance(model, LogisticRegression)
    assert model.coef_.all()
    assert len(model.coef_[0]) == 7

    # Include custom algorithm class hyperparameters (max_iter, n_jobs)
    model = sklearn_distance.model(
        lalonde_data,
        "logistic",
        max_iter=200,
        n_jobs=1,
    )

    assert isinstance(model, LogisticRegression)
    assert model.coef_.all()
    assert len(model.coef_[0]) == 7

    params = model.get_params()
    assert params["max_iter"] == 200
    assert params["n_jobs"] == 1


def test_propensity_algorithms(sklearn_distance, lalonde_data):
    """Algorithm arguments names return valid classifier model estimators"""
    propensity_algorithms = SKLEARN_DISTANCE_ALGORITHMS["propensity"]

    for algorithm, module_class in propensity_algorithms.items():
        module_name, class_name = module_class.rsplit(".", 1)
        assert getattr(import_module(module_name), class_name)

        model = sklearn_distance.model(lalonde_data, algorithm)
        assert isinstance(model, BaseEstimator)


def test_propensities(
    sklearn_distance, lalonde_logistic_model, lalonde_data
):
    """Propensity score arrays for target and non-target observations"""
    propensities = sklearn_distance.propensities(
        lalonde_data,
        model=lalonde_logistic_model,
    )

    target_count = len(lalonde_data[lalonde_data.iloc[:, 0] == 1])
    non_target_count = len(lalonde_data[lalonde_data.iloc[:, 0] == 0])

    assert (
        len(propensities["target"]) + len(propensities["non_target"])
        == lalonde_data.shape[0]
    )

    propensities = sklearn_distance.propensities(
        lalonde_data,
        "logistic",
    )

    assert len(propensities["target"]) == target_count
    assert len(propensities["non_target"]) == non_target_count

    assert 0 <= propensities["target"].all() <= 1
    assert 0 <= propensities["non_target"].all() <= 1


def test_propensity_distances(
    sklearn_distance, lalonde_data, lalonde_columns, lalonde_logistic_model
):
    """Pairwise m x n (target x non_target) differences between propensities"""
    propensities = sklearn_distance.propensities(
        lalonde_data,
        model=lalonde_logistic_model,
    )

    target_count = len(lalonde_data[lalonde_data.iloc[:, 0] == 1])
    non_target_count = len(lalonde_data[lalonde_data.iloc[:, 0] == 0])

    distances = sklearn_distance.propensity_distances(propensities=propensities)

    assert distances.shape == (target_count, non_target_count)
    assert 0 <= distances.all() <= 1

    distances = sklearn_distance.propensity_distances(
        lalonde_data,
        "logistic",
    )

    assert distances.shape == (target_count, non_target_count)
    assert 0 <= distances.all() <= 1


def test_covariate_distances(sklearn_distance, lalonde_data):
    """Non propensity covariate distance calculations"""
    distances = sklearn_distance.covariate_distances(
        lalonde_data,
        "mahalanobis",
    )
    assert distances.shape == (297, 425)

    distances = sklearn_distance.covariate_distances(
        lalonde_data,
        "euclidean",
    )
    assert distances.shape == (297, 425)


def test_distances(sklearn_distance, lalonde_data):
    """General distance calculation method that handles all algorithms"""
    distances = sklearn_distance.distances(
        lalonde_data,
        algorithm="logistic",
    )
    assert distances.shape == (297, 425)

    distances = sklearn_distance.distances(
        lalonde_data,
        algorithm="mahalanobis",
    )
    assert distances.shape == (297, 425)
