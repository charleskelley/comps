"""
Facade interface for matching target and non-target observations.
"""
from typing import Optional, Sequence, Union

from pandas import PandasDataFrame
from pyspark.sql import DataFrame as SparkSQLDataFrame

from comps.match.distance import Distance


class Match:
    """
    Class used for all target and non-target observation matching
    processes.

    Attributes:
        backend: The computational backend to use.

        data: DataFrame input with all observation data that will be used to
            drive the target to non-target matching process and must include
            all features (numeric only) and a binary target identification
            variable.

        distance: A ``comps.match.distance.facade.Distance`` class instance
            that provides standardized access to all target vs non-target
            propensity and covariate distance calculation algorithms that can
            be used to calculate the distance between target and non-target
            observations and summary information for the last distance
            calculation made.

        matches: Target to non-target ID, or data index, matches based on the
            data provided as an input and whether an ID variable was included
            in the data. If an ID variable is used, ID to ID matches will be
            provided, otherwise index to index matches from the provided data
            input will used.

        selector: A ``comps.match.selector.facade.Selector`` class instance
            that provides standardized access to all target vs non-target
            match selection methods that can be used to select target to
            non-target matches based on a distance matrix.

        vignette: Dictionary summary of all the key data input, distance
            calculation, and match selection properties that produced the
            observation matches.
    """

    def __init__(self):
        self.distance = Distance()

    def calculate_distances(
        self,
        data: Union[PandasDataFrame, SparkSQLDataFrame],
        distance: str = "logistic",
        selector: str = "nearest",
        features: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
        id: Optional[str] = None,
        engine: str = "sklearn",
        **kwargs,
    ):
        """
        Args:
            data: DataFrame input with all observation data that will be used to
                drive the target to non-target matching process and must include
                all features (numeric only) and a binary target identification
                variable.
        """

        pass

    def select_matches(self):
        pass

    def match_observations(self):
        pass

    def balance_summary(self):
        pass

    def distance_summary(self):
        pass

    def selector_summary(self):
        pass

    def vignette_summary(self):
        pass

    def write_report(self):
        pass
