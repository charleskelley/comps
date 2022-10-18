import pytest

from comps.match.distance.base import Distance


@pytest.fixture
def distance():
    """Initialized instance of SklearnDistance class Engine"""
    distance_instance = Distance()

    return distance_instance


def test_class_rows(distance, lalonde_pandas):
    """Identifying target and non-target observation row numbers"""
    rows = distance._class_rows(lalonde_pandas, "treatment")

    assert len(rows["target"]) == 297
    assert (min(rows["target"]), max(rows["target"])) == (0, 296)

    assert len(rows["non_target"]) == 425
    assert (min(rows["non_target"]), max(rows["non_target"])) == (297, 721)


def test_class_uids(distance, lalonde_pandas):
    """Identifying target and non-target observation row numbers"""
    uids = distance._class_uids(lalonde_pandas, "treatment", "observation")

    assert uids["name"] == "observation"

    assert len(uids["target"]) == 297
    assert (min(uids["target"]), max(uids["target"])) == (1, 297)

    assert len(uids["non_target"]) == 425
    assert (min(uids["non_target"]), max(uids["non_target"])) == (298, 722)


def test_features(distance, lalonde_pandas, lalonde_columns):
    """Default identification of undeclared column names from DataFrame"""
    with pytest.raises(TypeError):
        distance._features(lalonde_pandas, lalonde_columns["features"])

    features_passed_through = distance._features(
        lalonde_pandas, lalonde_columns["target"], lalonde_columns["features"]
    )
    assert features_passed_through == lalonde_columns["features"]

    expected_default_features = ["dataset"] + lalonde_columns["features"] + ["re78"]
    default_features = distance._features(
        lalonde_pandas,
        lalonde_columns["target"],
        uid=lalonde_columns["uid"],
    )
    assert default_features == expected_default_features
