"""
Observation distance algorithms implemented using Spark.
"""
from typing import Optional, TypeAlias, Union

from pyspark import SparkConf
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    RandomForestClassifier,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import size


SparkPropensityClassifier: TypeAlias = Union[
    GBTClassifier,
    MultilayerPerceptronClassifier,
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
]


SPARK_DISTANCE_ALGORITHMS = {
    "propensity": {
        "boosted_tree": "pyspark.ml.classification.GBTClassifier",
        "decision_tree": "pyspark.ml.classification.DecisionTreeClassifier",
        "logistic": "pyspark.ml.classification.LogisticRegression",
        "neural_network": "pyspark.ml.classification.MultilayerPerceptronClassifier",
        "random_forest": "pyspark.ml.classification.RandomForestClassifier",
    },
}


class SparkDistance:
    """
    Class for making propensity score and covariate distance calculations based
    on PySpark SQL DataFrame inputs using the Spark MLlib framework.

    Attributes:
        classifiers: Dictionary of algorithm names mapped to MLlib classifier
            classes that can be used for propensity score modeling to predict
            scores used to calculate propensity score distance between target
            and non-target class observations.

        conf: SparkConf object if one was used to build the PySpark SQL
            SparkSession.

        session: PySpark SQL SparkSession reference.
    """

    def __init__(self) -> None:
        self.classifiers: dict = {
            "boosted_tree": GBTClassifier,
            "decision_tree": DecisionTreeClassifier,
            "logistic": LogisticRegression,
            "neural_network": MultilayerPerceptronClassifier,
            "random_forest": RandomForestClassifier,
        }
        self.conf: Optional[SparkConf] = None
        self.session: SparkSession = SparkSession.builder.getOrCreate()

    def _default_layers(
        self, data: DataFrame, features_column: str = "features"
    ) -> list[int]:
        """Calculate layers hyperparameter for MultilayerPerceptron"""
        first_row = data.first()
        features_nodes = first_row[features_column].size
        output_nodes = 2

        if features_nodes > 20:
            return [
                features_nodes,
                round(features_nodes * 0.66),
                round(features_nodes * 0.33),
                output_nodes,
            ]

        return [
            features_nodes,
            round(features_nodes * 0.5),
            output_nodes,
        ]

    def model(
        self,
        data: DataFrame,
        algorithm: str,
        **kwargs,
    ) -> SparkPropensityClassifier:
        """
        Fit a classifier algorithm to the data to create a model instance that
        can be used to calculate propensity scores.

        Args:
            data: Data input with at least two columns, where one column is a
                binary target class indicator, and the second column is a
                vector of numeric features used to fit the classifier
                algorithm. The column names must also be provided using
                ``**kwargs``. Specifically, using the ``labelCol`` and and
                ``featuresCol`` keyword arguments.

                It is also possible to provide additional columns and data such
                as observation sample weights using the ``weightCol`` keyword
                argurment. See the Spark Python API documentation for details.

            algorithm: Name of the algorithm to fit to the data to create a
                fitted classifier model object.

            **kwargs: Keyword arguments to pass through to the MLlib classifier
                algorithm to set hyperparameters prior to fitting. See the
                Spark Python API docs for the classifier algorithm used.

        Returns:
            Fitted Spark classifier model Transformer instance.
        """
        classifier = self.classifiers[algorithm]

        if algorithm == "neural_network":
            if not kwargs:
                kwargs = {}

            features_column = (
                kwargs["featuresCol"] if "featuresCol" in kwargs else "features"
            )
            if "layers" not in kwargs:
                kwargs["layers"] = self._default_layers(data, features_column)

        fitted_model = classifier(**kwargs).fit(data)

        return fitted_model

    def distances(
        self,
        data: DataFrame,
        algorithm: str = "logistic",
        **kwargs,
    ) -> DataFrame:
        """
        Calculate the pairwise distance between each target observation and all
        non-target observations and create an ndarray matrix where each target
        observation i represented as a row and each non-target observation j is
        represented as a column and the value at row i, column j is the distance
        between target observation i and non-target observation j.

        Args:
            data: DataFrame input with all observation data.

            algorithm: A valid propensity score distance classifier or
                covariate distance algorithm name.

            propensities: Whether the pairwise propensities matrix should be
                retained in the Distances return object in addition to the
                distance values for classifier algorthms.

            **kwargs: Keyword arguments to pass through to the classification
                algorithm class initializer for the configured compute engine
                or to the scikit learn pairwise_distances function if the
                algorithm is a covariate distance metric.

        Returns:
            Distances class instance with key details from distance calculation
            scenario and model if available.
        """
        pass
