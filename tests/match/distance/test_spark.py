from importlib import import_module

import pytest
from pyspark.ml.base import Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from comps.match.distance.spark import SPARK_DISTANCE_ALGORITHMS, SparkDistance


pystestmark = pytest.mark.unit


@pytest.fixture
def spark_distance():
    """Initialized instance of SparkDistance class Engine"""
    spark_engine = SparkDistance()

    return spark_engine


@pytest.fixture
def lalonde_model_data(lalonde_spark, lalonde_columns):
    """Lalonde Spark data ready for distance calculation methods"""
    label_column = lalonde_columns["target"]
    feature_columns = lalonde_columns["features"]
    model_columns = [label_column] + feature_columns

    model_data = lalonde_spark[model_columns]

    for column in model_data.columns:
        model_data = lalonde_spark.withColumn(
            column, lalonde_spark[column].cast("float")
        )

    vector_assembler = VectorAssembler(outputCol="features")
    vector_assembler.setInputCols(feature_columns)
    model_data = vector_assembler.transform(model_data)

    return model_data


@pytest.fixture
def lalonde_logistic_model(lalonde_model_data, lalonde_columns):
    """Default scikit-learn Logistic regression model fit to Lalonde data"""
    label_column = lalonde_columns["target"]
 
    return LogisticRegression(labelCol=label_column).fit(lalonde_model_data)


def test_spark_distance_init(spark_distance):
    """Initialization of SparkDistance class and SparkSession discovery"""
    assert isinstance(spark_distance, SparkDistance)
    assert isinstance(spark_distance.session, SparkSession)


def test_default_layers(spark_distance, lalonde_model_data):
    """Create a list of layers for a multilayer perceptron"""
    expected_layers = [7, 4, 2]
    assert spark_distance._default_layers(lalonde_model_data) == expected_layers


def test_model(spark_distance, lalonde_model_data):
    """Fitting of Spark classifier model"""
    model = spark_distance.model(lalonde_model_data, "logistic", labelCol="treatment")

    assert isinstance(model, Transformer)


def test_propensity_algorithms(spark_distance, lalonde_model_data):
    """Algorithm arguments names return valid classifier model estimators"""
    propensity_algorithms = SPARK_DISTANCE_ALGORITHMS["propensity"]

    for algorithm, module_class in propensity_algorithms.items():
        module_name, class_name = module_class.rsplit(".", 1)
        assert getattr(import_module(module_name), class_name)

        model = spark_distance.model(
            lalonde_model_data, algorithm, labelCol="treatment"
        )
        assert isinstance(model, Transformer)
