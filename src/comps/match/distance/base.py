"""
Base class for calculating covariate distance between observations.
"""
import functools
from typing import Any, Optional, Sequence, Union

from numpy import number
from numpy.typing import NDArray
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkSQLDataFrame

from comps.match.distance.engine import (Engine, SklearnPropensityClassifier,
                                         SparkPropensityClassifier)
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
        engine: Reference to the last data input provided when a distance
            calculation was made.

        engines: Reference to the last data input provided when a distance
            calculation was made.
    """

    def __init__(self) -> None:
        self._engine: Union[SklearnDistance, SparkDistance] = SklearnDistance()
        self._engines: dict[str, Union[SklearnDistance, SparkDistance]] = {
            "sklearn": self._engine
        }

    # Combined engines and engine properties allows for easy exchange of engine
    # instances at runtime without re-instantiating an engine instance and losing
    # state. Actual engine exchange is handled via the @check_engine decorator.
    @property
    def engines(self) -> dict[str, Union[SklearnDistance, SparkDistance]]:
        return self._engines

    @engines.setter
    def engines(self, engine: Union[SklearnDistance, SparkDistance]) -> None:
        if isinstance(engine, SparkDistance) and "spark" not in self.engines:
            self._engines["spark"] = engine

    @property
    def engine(self) -> Union[SklearnDistance, SparkDistance]:
        return self._engine

    @engine.setter
    def engine(self, engine_name: str) -> None:
        if engine_name == "sklearn":
            self._engine = self.engines["sklearn"]

        elif engine_name == "spark":
            if "spark" not in self.engines:
                self._engine = SparkDistance()
            else:
                self._engine = self.engines["spark"]

    @check_engine
    def model(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        algorithm: str,
        **kwargs,
    ) -> Union[SklearnPropensityClassifier, SparkPropensityClassifier]:
        """
        Fit a classifier algorithm to the data to create a model instance that
        can be used to calculate propensity scores.

        Args:
            algorithm: Name of the algorithm to train to create the new model
                object.

            data: DataFrame input with all observation data that will be used to
                train the logistic regression model.

            target: Name of column in data that has numeric binary indicator
                for target observations where 1 indicates observations belong to
                the target class that the logistic regression model will be fit
                to predict probability for.

            features: List of column names to specify the columns used as input
                features for model fitting. If a list of feature names is not
                provided, all column names in the input data except for the
                target column are assumed to be features.

            **kwargs: Keyword arguments to pass through to the logistic
                regression model trainer for the configured compute engine.

        Returns:
            Fitted scikit-learn classifier model instance.
        """
        return self.engine.model(data, algorithm, **kwargs)
