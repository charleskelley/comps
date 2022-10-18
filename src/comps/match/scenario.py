"""
Distances return object class for covariate distance calculations.
"""
from dataclasses import dataclass
from typing import Optional, TypeAlias, Union

from numpy import arange, array, column_stack, meshgrid, ndarray, number
from numpy.typing import NDArray
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkSQLDataFrame
from pyspark.ml.classification import GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.classification import DecisionTreeClassifier as SparkDecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


SklearnClassifier: TypeAlias = Union[
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
]


SparkClassifier: TypeAlias = Union[
    GBTClassifier,
    MultilayerPerceptronClassifier,
    SparkDecisionTreeClassifier,
    SparkLogisticRegression,
    SparkRandomForestClassifier,
]


@dataclass
class Scenario:
    """
    Dataclass to hold all data and metadata from a target to non-target
    observation matching scenario including details on any distance calculations
    made and matching methods used.

    Attributes:
        algorithm: The classifier algorithm used for propensity scoring or a
            covariate distance algorithm name argument for the distances method.

        data: Reference to the data used for matching target and non-target
            observations. Must include a binary target class indicator column
            and any feature source data used for distance calculations and
            or required as input to the matching method used.

        distances: Matrix of pairwise distances between each target observation
            and all non-target observations with each target observation i
            represented as a row and each non-target observation j is
            represented as a column and the value at row i, column j is the
            distance between target observation i and non-target observation j.

        target: Name of column in data that is the numeric binary indicator
            for target observations where 1 indicates observations belong to the
            target class. For propensity score distance classifier algorithms
            model, the model is fit to predict probability of an observation
            belonging to the target class.

        features: List of column names to specify the columns used as input
            features for model fitting or covariate distance calculation.

        model: Fitted classifier model instance if a propensity score distances
            algorithm was used.
    """

    algorithm: str
    data: Union[PandasDataFrame, SparkSQLDataFrame]
    values: Union[NDArray[number], SparkSQLDataFrame]
    rows: dict[str, NDArray[number]]
    target: str
    features: list[str]
    model: Optional[Union[SklearnClassifier, SparkClassifier]] = None
    propensities: Optional[Union[dict[str, NDArray[number]], SparkSQLDataFrame]] = None
    uids: Optional[dict[str, Union[str, NDArray]]] = None

    def _data_row_pairs(self) -> NDArray[number]:
        """
        Cartesian cross of target and non-target data row number 1D arrays into
        a two column array (target, non-target) of target and non-target
        observation row numbers from the data input to the distances
        calculation scenario.
        """
        if not self.rows:
            raise ValueError("No 'rows' attribute available")

        data_row_pairs = array(
            meshgrid(self.rows["target"], self.rows["non_target"])
        ).T.reshape(-1, 2)

        return data_row_pairs

    def _rows_index_pairs(self) -> NDArray[number]:
        """
        Cartesian cross of target and non-target array item indices in the
        'rows' attribute so that pairwise position can be looked up in the
        pairwise distance matrix m_target x n_non_target 'values' attribute.
        """
        rows_index_pairs = array(
            meshgrid(
                arange(len(self.rows["target"])), arange(len(self.rows["non_target"]))
            )
        ).T.reshape(-1, 2)

        return rows_index_pairs

    def _uid_pairs(self) -> NDArray[number]:
        """All unique ID target and non-target pairs"""
        if not self.uids:
            raise ValueError("No 'uids' attribute available")

        uid_pairs = array(
            meshgrid(self.uids["target"], self.uids["non_target"])
        ).T.reshape(-1, 2)

        return uid_pairs

    def _pair_distance(
            self, rows_index_pairs: NDArray[number]
    ) -> NDArray[number]:
        """Absolute distance between paired target and non-target observations"""
        assert isinstance(self.values, ndarray)
        pair_distance = self.values[
            rows_index_pairs[:, 0], rows_index_pairs[:, 1]
        ]  # type: ignore

        return pair_distance

    def _pair_propensities(
            self, rows_index_pairs: NDArray[number]
    ) -> NDArray[number]:
        """Propensity scores for target and non-target observations in pair"""
        if not self.propensities:
            raise ValueError("No 'propensities' attribute available")

        target_propensities = self.propensities["target"][
            rows_index_pairs[:, 0]
        ]
        non_target_propensities = self.propensities["non_target"][
            rows_index_pairs[:, 0]
        ]

        assert isinstance(target_propensities, ndarray)
        assert isinstance(non_target_propensities, ndarray)
        return column_stack(
            (target_propensities, non_target_propensities)
        )

    def _distances_pandas(self) -> PandasDataFrame:
        """Converts pairwise distances values into DataFrame with IDs"""
        if self.uids:
            assert isinstance(self.uids["name"], str)
            target_uid_name = "target_" + self.uids["name"]
            non_target_uid_name = "non_target_" + self.uids["name"]
        else:
            target_uid_name = "target_data_uid"
            non_target_uid_name = "non_target_data_uid"

        rows_index_pairs = self._rows_index_pairs()
        uid_pairs = (
            self._uid_pairs() if self.uids else self._data_row_pairs()
        )
        pair_distance = self._pair_distance(rows_index_pairs)

        distances = {
            target_uid_name: uid_pairs[:, 0],
            non_target_uid_name: uid_pairs[:, 1],
            "target_pair": rows_index_pairs[:, 0] + 1,
            "distance": pair_distance,
        }

        if self.propensities:
            pair_propensities = self._pair_propensities(
                rows_index_pairs
            )
            propensities = {
                "target_propensity": pair_propensities[:, 0],
                "non_target_propensity": pair_propensities[:, 1],
            }
            distances_with_propensities = distances | propensities

            return PandasDataFrame(distances_with_propensities)

        return PandasDataFrame(distances)

    def dataframe(self) -> Union[PandasDataFrame, SparkSQLDataFrame]:
        """
        Normalized DataFrame with all pairwise distances calculated between
        target and non-target observations and propensity scores if available.
        """
        if isinstance(self.values, ndarray):
            return self._distances_pandas()

        raise ValueError("No valid 'values' attribute to return data for")

    def summary(self) -> Union[PandasDataFrame, SparkSQLDataFrame]:
        """
        Normalized DataFrame with all pairwise distances calculated between
        target and non-target observations and propensity scores if available.
        """
        pass
