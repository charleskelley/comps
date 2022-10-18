"""
Abstract base class for distance calculation engine interface.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeAlias, Union

from numpy import number
from numpy.typing import NDArray
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkSQLDataFrame
from pyspark.ml.linalg import Vector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.classification import DecisionTreeClassifier as SparkDecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier

from comps.match.distance.distances import Distances, SklearnClassifier, SparkClassifier


SklearnPropensityClassifier: TypeAlias = Union[
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
]


SparkPropensityClassifier: TypeAlias = Union[
    GBTClassifier,
    MultilayerPerceptronClassifier,
    SparkDecisionTreeClassifier,
    SparkLogisticRegression,
    SparkRandomForestClassifier,
]


class Engine(ABC):
    """
    Abstract base class defining a standardized user interface for making
    propensity or covariate distance calculations using different
    computational frameworks or engines.
    """

    @abstractmethod
    def model(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        algorithm: Optional[str] = None,
        sample_weight: Optional[NDArray[number]] = None,
        **kwargs: Any,
    ) -> Union[SklearnPropensityClassifier, SklearnPropensityClassifier]:
        pass

    @abstractmethod
    def propensities(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        algorithm: Optional[str] = None,
        model: Optional[Union[PandasDataFrame, SparkSQLDataFrame]] = None,
        **kwargs: Any,
    ) -> dict[str, Union[NDArray[number], Vector]]:
        pass

    @abstractmethod
    def propensity_distances(
            self,
            data: Union[PandasDataFrame, SparkSQLDataFrame],
            algorithm: Optional[str] = None,
            model: Optional[Union[SklearnClassifier, SparkClassifier]] = None,
            propensities: Optional[dict[str, Union[NDArray[number], Vector]]] = None,
            **kwargs: Any,
    ) -> NDArray[number]:
        pass

    @abstractmethod
    def covariate_distances(
            self,
            data: Union[PandasDataFrame, SparkSQLDataFrame],
            algorithm: Optional[str] = None,
            model: Optional[Union[SklearnClassifier, SparkClassifier]] = None,
            **kwargs: Any,
    ) -> NDArray[number]:
        pass

    @abstractmethod
    def distances(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        algorithm: str = "logistic",
        **kwargs: Any,
    ) -> NDArray[number]:
        pass
