"""
Distance calculation methods implemented using Spark.
"""
from typing import Optional

from pyspark import SparkConf
from pyspark.sql import DataFrame, SparkSession

from comps.match.distance.engine import Engine


class SparkDistance(Engine):
    conf: Optional[SparkConf] = None
    session: Optional[SparkSession] = None

    def build_session(self, conf: Optional[SparkConf] = None):
        self.session = (
            SparkSession.builder.config(conf=conf).getOrCreate()
            if conf
            else SparkSession.builder.getOrCreate()
        )

    def logistic(
        self,
        data: DataFrame,
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> DataFrame:
        raise NotImplementedError

    def boosted_tree(
        self,
        data: DataFrame,
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> DataFrame:
        raise NotImplementedError

    def partition_tree(
        self,
        data: DataFrame,
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> DataFrame:
        raise NotImplementedError

    def random_forest(
        self,
        data: DataFrame,
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> DataFrame:
        raise NotImplementedError

    def neural_network(
        self,
        data: DataFrame,
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> DataFrame:
        raise NotImplementedError

    def naive_bayes(
        self,
        data: DataFrame,
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> DataFrame:
        raise NotImplementedError

    def covariate(
        self,
        data: DataFrame,
        target: str,
        id: Optional[str] = None,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> DataFrame:
        raise NotImplementedError
