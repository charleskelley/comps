"""
Base class for calculating covariate distance between observations.
"""
import functools
from typing import Any, Optional, Sequence, Union

from numpy import number
from numpy.typing import NDArray
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkSQLDataFrame

from comps.match.distance.engine import Engine
from comps.match.distance.sklearn import SklearnDistance
from comps.match.distance.spark import SparkDistance


def check_engine(func):
    """
    Decorator to set Distance class engine property to proper value based on
    type of data passed to method.
    """

    @functools.wraps(func)
    def checked_engine_func(*args, **kwargs):
        _self = args[0]

        if isinstance(args[1], PandasDataFrame):
            _self.engine = "sklearn"

        elif isinstance(args[1], SparkSQLDataFrame):
            _self.engine = "spark"

        return func(*args, **kwargs)

    return checked_engine_func


class Distance:
    """
    Standardized interface for calculating covariate distance between
    observations using various methods.

    MatchIt Distance Methods Implemented

    |x| = Implemented

    Propensity Score

    |x| 1. Generalized linear model (glm)
           * dask - dask_ml.linear_model.LogisticRegression
           * pyspark - pyspark.ml.classification.LogisticRegression (also lasso, ridge, elasticnet)
           * sklearn - sklearn.linear_model.LogisticRegression (also lasso, ridge, elasticnet)

        2. Generalized additive model (gam)

        3. Generalized boosted model (gbm)
           * pyspark - pyspark.ml.classification.GBTClassifier
           * sklearn - sklearn.ensemble.GradientBoostingClassifier

    |x| 4. Lasso, ridge, or elasticnet (lasso, ridge, elasticnet) => See 1.

        5. Classification tree (rpart)
           * pyspark - pyspark.ml.classification.DecisionTreeClassifier
           * sklearn - sklearn.tree.DecisionTreeClassifier

    |x| 6. Random Forest classification (randomforest)
        7. Neural Network (nnet) - single-layer hidden network
           - pyspark - pyspark.ml.classification.MultilayerPerceptronClassifier
           - sklearn - sklearn.neural_network.MLPClassifier

        8. Covariate balancing propensity scores (cbps) -
        9. Bayesian additive regression trees (bart) -
           * dask - dask_ml.naive_bayes.GaussianNB
           * pyspark - pyspark.ml.classification.NaiveBayes (GaussianNB, BernouliNB, MiultnomialNB)
           * sklearn - sklearn.naive_bayes (GaussianNB, BernouliNB, MiultnomialNB)

    Covariate Distances

    |x| 1. Euclidean (euclidean)
    |x| 2. Scaled euclidean (scaled_euclidian)
    |x| 3. Mahalanobis distance (mahalanobis)
        4. Robust Mahalanobis distance (robust_mahalanobis)

    Attributes:
        data: Reference to the last data input provided when a distance
            calculation was made.

        features: List of features from the last data input distance
            calculation performed.

        id: If available, the name of the observation ID column in the last
            data input provided when a distance calculation was made.

        method: Name of the distance calculation method last used.

        model: A trained model object for propensity distance calculation
            methods that train a model object and use it to make target class
            propensity predictions.

        propensities: Summary of the propensity scores for all target and
            non-target observations prior to the pairwise distance calculation
            between each target observation and all non-target observations.

        target: If available, the name of binary indicator variable in the last
            data input provided when a propensity score distance calculation was
            made.
    """

    def __init__(self) -> None:
        self._engine: Engine = SklearnDistance()
        self._engines: dict[str, Engine] = {"sklearn": self._engine}
        self.data: Optional[Union[PandasDataFrame, SparkSQLDataFrame]] = None
        self.features: Optional[Sequence[str]] = None
        self.id: Optional[str] = None
        self.method: Optional[str] = None
        self.model: Optional[Any] = None
        self.propensities: Optional[
            Union[NDArray[number], PandasDataFrame, SparkSQLDataFrame]
        ] = None
        self.target: Optional[str] = None

    # Combined engines and engine properties allows for easy exchange of engine
    # instances at runtime without reinstantiating an engine instance and losing
    # state. Actual engine exchange is handled via the @check_engine decorator
    @property
    def engines(self) -> dict[str, Engine]:
        return self._engines

    @engines.setter
    def engines(self, engine: Engine) -> None:
        if isinstance(engine, SparkDistance) and "spark" not in self.engines:
            self._engines["spark"] = engine

    @property
    def engine(self) -> Engine:
        return self._engine

    @engine.setter
    def engine(self, engine_name: str) -> None:
        if engine_name == "sklearn":
            self._engine = self.engines["sklearn"]

        elif engine_name == "spark":
            if "spark" not in self.engines:
                self.engines = SparkDistance()

            self._engine = self.engines["spark"]

    @check_engine
    def logistic(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> Union[NDArray[number], PandasDataFrame, SparkSQLDataFrame]:
        """
        Propensity score distance estimation using logistic regression. Train a
        a logistic regressor using the provided features and target.

        Args:
            data: Data table input with observation tracking variables, a
                binary target variable, and all features (numeric only) that
                will be used to train the logistic regressor. If the the names
                of the feature columns are are not specified using the features
                argument, it is assumed all columns not specified by the target
                and id arguments are features and will be used for model
                training.

            target: Name of the binary target variable column in data that
                represents the class the logistic regressor will be trained to
                predict.

            features: Names of feature variables in the data to use for
                training the logistic regressor.

            **kwargs: Additional keyword arguments to set hyperparameters for
                configuring the estimator model prior to training.

        Returns:
            Pairwise absolute differences between propensity scores for each
            target observation and all non-target observations.
        """
        return self.engine.logistic(data, target, id, features, **kwargs)

    def boosted_tree(self):
        pass

    def partition_tree(self):
        pass

    def random_forest(self):
        pass

    def neural_network(self):
        pass

    def naive_bayes(self):
        pass

    def covariate(self):
        pass

    def _set_call_attributes(
        self,
        data: Optional[Union[PandasDataFrame, SparkSQLDataFrame]] = None,
        features: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
        id: Optional[str] = None,
        method: Optional[str] = None,
    ):
        """Set the attributes always defined at call time"""
        self.data = data
        self.features = features
        self.target = target
        self.id = id
        self.method = method

    def calculate_distances(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        features: Optional[list[str]] = None,
        target: Optional[str] = None,
        id: Optional[str] = None,
        method: str = "logistic",
        **kwargs,
    ) -> Union[PandasDataFrame, SparkSQLDataFrame]:
        """
        Args:
            data: Data input with all observation features will be used to
                drive the target to non-target matching process and must include
                all features (numeric only) and a binary target identification
                variable for propensity score distance calculation methods.

            features:

            target:

            id: Name of variable column in data that is has an ID value that
                uniquely identifies every observation. It is up to the user to
                guarantee uniqueness, it will not be checked.

            method:

            **kwargs:

        Returns:
            A Pandas or PySpark DataFrame
        """
        self._set_call_attributes(data, features, target, id, method)

        return PandasDataFrame()
