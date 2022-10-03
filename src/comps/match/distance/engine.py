"""
Abstract base class for distance calculation engine interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeAlias, Union

from numpy import number
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame as PandasDataFrame
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    RandomForestClassifier,
)
from pyspark.sql import DataFrame as SparkSQLDataFrame
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
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    RandomForestClassifier,
]


@dataclass
class Distances:
    """
    Standardized dataclass output object to hold all data from a propenisty
    score or covariate distance scenario from the distance calculation engine
    methods.

    Attributes:
        distances:
        features:
        method:
        target:
        model:
        propensities:
        rows:
        uids:
    """

    distances: Union[NDArray[number], SparkSQLDataFrame]
    target: str
    features: list[str]
    algorithm: str
    model: Optional[Union[SklearnClassifier, SparkClassifier]]
    propensities: Optional[Union[dict[str, NDArray[number]], SparkSQLDataFrame]]
    rows: Optional[dict[str, ArrayLike]]
    uids: Optional[dict[str, Union[str, ArrayLike]]]

    def _distances_dataframe(self) -> Union[PandasDataFrame, SparkSQLDataFrame]:
        if self.uids:
            assert isinstance(self.uids["name"], str)
            target_uid = ("target_" + self.uids["name"],)
            non_target_uid = "non_target_" + self.uids["name"]
        else:
            target_uid = "target_uid"
            non_target_uid = "non_target_uid"

        distances_dict = {
            target_uid: None,
            non_target_uid: None,
            "target_pair": None,
            "distance": None,
            "target_propensity": None,
            "non_target_propensity": None,
        }
        # if isinstance(self.distances, ndarray):

        return PandasDataFrame(distances_dict)

    def dataframe(self) -> Union[PandasDataFrame, SparkSQLDataFrame]:
        """
        Normalized DataFrame with all pairwise distances calculated between
        target and non-target observations and propensity scores if available.
        """
        pass

    def summary(self) -> Union[PandasDataFrame, SparkSQLDataFrame]:
        """
        Normalized DataFrame with all pairwise distances calculated between
        target and non-target observations and propensity scores if available.
        """
        pass


class Engine(ABC):
    """
    Abstract base class defining a standardized user interface for making
    propensity or covariate distance calculations using different
    computational frameworks or engines.

    Attributes:
        data: Reference to the last DataFrame input with all observation data
            used for the last model that was fit or covariate distance
            calculation made.

        models: Dictionary of distance calculation method (model) names mapped
            to the last fitted model of. This allows multiple models to easily
            be fit on the same data for comparison.
    """

    data: Optional[Union[PandasDataFrame, SparkSQLDataFrame]] = None
    models: dict[str, Union[SklearnClassifier, SparkClassifier]] = {}

    @abstractmethod
    def fit(
        self,
        algorithm: str,
        data: Optional[Union[PandasDataFrame, SparkSQLDataFrame]],
        target: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> Union[SklearnClassifier, SparkClassifier]:
        pass

    @abstractmethod
    def calculate(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        target: str,
        features: Optional[list[str]] = None,
        uid: Optional[str] = None,
        algorithm: str = "logistic",
        **kwargs,
    ) -> Distances:
        pass
