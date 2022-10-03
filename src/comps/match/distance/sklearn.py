"""
Distance calculation methods implemented using scikit-learn.
"""
from typing import Optional, Union

from numpy import number, reshape, where
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from comps.match.distance.engine import Distances, Engine, SklearnClassifier


class SklearnDistance(Engine):
    """
    Class for calculating covariate distance calculations based on Pandas
    DataFrame inputs using the scikit-learn framework.
    """

    algorithms = {
        "boosted_tree": GradientBoostingClassifier,
        "decision_tree": DecisionTreeClassifier,
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "neural_network": MLPClassifier,
    }

    def _class_rows(self, data: DataFrame, target: str) -> dict[str, ArrayLike]:
        """Binary target DataFrame row target binary row number tracking dictionary"""
        target_rows = where([data[target] == 1])[1]
        non_target_rows = where([data[target] == 0])[1]

        return {"target": target_rows, "non_target": non_target_rows}

    def _class_uids(
        self,
        data: DataFrame,
        target: str,
        uid: str,
    ) -> dict[str, Union[str, ArrayLike]]:
        """
        Capture unique IDs in effiicient data structure for tracking alongside
        class rows.
        """
        target_uids = data[data[target] == 1][uid]
        non_target_uids = data[data[target] == 0][uid]

        return {"name": uid, "target": target_uids, "non_target": non_target_uids}

    def _class_propensities(
        self,
        model: SklearnClassifier,
        data: DataFrame,
        target: str,
        features: list[str],
    ) -> dict[str, NDArray[number]]:
        """
        Dictionary of bundled target class probabilities for target and
        non-target observations in data.
        """
        target_probabilities = model.predict_proba(  # type: ignore
            data.query(f"{target} == 1")[features].to_numpy(float)
        )[
            :, 1
        ]  # type: NDArray[number]

        non_target_probabilities = model.predict_proba(  # type: ignore
            data.query(f"{target} == 0")[features].to_numpy(float)
        )[
            :, 1
        ]  # type: NDArray[number]

        return {"target": target_probabilities, "non_target": non_target_probabilities}

    def _propensity_distances(
        self,
        target_class_propensities: dict[str, NDArray[number]],
    ) -> NDArray[number]:
        """
        Pairwise m x n (target x non-target) ndarray of absolute differences
        between target and non-target observation propensity scores.
        """
        target_propensities = reshape(
            target_class_propensities["target"],
            (len(target_class_propensities["target"]), 1),
        )  # type: NDArray[number]

        non_target_propensities = reshape(
            target_class_propensities["non_target"],
            (1, len(target_class_propensities["non_target"])),
        )  # type: NDArray[number]

        return abs(target_propensities - non_target_propensities)

    def _features(
        self,
        data: DataFrame,
        target: str,
        features: Optional[list[str]] = None,
        uid: Optional[str] = None,
    ) -> list[str]:
        """Isolate default DataFrame feature column names if not provided"""
        if features is None:
            features = [
                column for column in list(data.columns) if column not in {target, uid}
            ]

        return features

    def fit(
        self,
        algorithm: str,
        data: DataFrame,
        target: str,
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Fit a classifier algorithm to the data to create a model instance that
        can be used to calculate propensity scores. The trained model is stored
        in the models attribute dictionary mapped to a key with the same name as
        the algorithm argument provided.

        Args:
            algorithm: Name of the algorithm to train to create the new model
                object.

            data: DataFrame input with all observation data that will be used to
                train the logistic regression model.

            target: Name of column in data that has numeric binary indicator
                for target observations where 1 indicates indicate observations
                belong to the target class that the logistic regression model
                will be fit to predict probability for.

            features: List of column names to specify the columns used as input
                features for model fitting. If a list of feature names is not
                provided, all column names in the input data except for the
                target column are assumed to be features.

            **kwargs: Keyword arguments to pass through to the logistic regression
                model trainer for the configured compute engine.

        """
        self.data = data
        self.models[algorithm] = self.algorithms[algorithm](**kwargs).fit(
            data[features].astype(float), data[target].astype(float)
        )

    def logistic(
        self,
        data: DataFrame,
        target: str,
        features: list[str],
        uid: Optional[str] = None,
        retain_propensities: bool = False,
        **kwargs,
    ) -> Distances:
        """
        Train a logistic regression model and use the model to make target
        class probability predictions for both the target and non-target
        observations in the training data.

        Args:
            data: DataFrame input with all observation data that will be used to
                train the logistic regression model.

            target: Name of column in data that has numeric binary indicator
                for target observations where 1 indicates indicate observations
                belong to the target class that the logistic regression model
                will be fit to predict probability for.

            features: List of column names to specify the columns used as input
                features for model fitting. If a list of feature names is not
                provided, all column names in the input data except for the
                target column are assumed to be features.

            **kwargs: Keyword arguments to pass through to the logistic regression
                model trainer for the configured compute engine.

            uid: Name of unique ID column in data that uniquely identifies
                individual observations or rows in data and should be used to
                identify observations instead of an observation's index
                location in data.

        Returns:
            Distances class instance with data from distance calculation scenario.
        """
        model = LogisticRegression(**kwargs).fit(
            data[features].to_numpy(float), data[target].to_numpy(int)
        )

        class_propensities = self._class_propensities(model, data, target, features)
        distances = self._propensity_distances(class_propensities)

        return Distances(
            distances,
            target,
            features,
            algorithm="logistic",
            model=model,
            propensities=class_propensities if retain_propensities else None,
            rows=self._class_rows(data, target),
            uids=self._class_uids(data, target, uid) if uid else None,
        )

    def calculate(
        self,
        data: DataFrame,
        target: str,
        features: Optional[list[str]] = None,
        algorithm: str = "logistic",
        uid: Optional[str] = None,
        **kwargs,
    ) -> Distances:
        raise NotImplementedError

    # def covariate(
    #     self, test_features, control_features, distance_metric: str = "mahalanobis", **kwargs
    # ) -> NDArray[number]:
    #     """
    #     Calculate the pairwise distance between each test observation and all
    #     control observations where the returned matrix has each observation
    #     represented as a row and each control observation represented as a column
    #     an the value at row i, column j is the distance between test observation i
    #     and control observation j.

    #     This function is a wrapper for using the ``sklearn.metrics.pairwise_distances``
    #     function to calculate pairwise distance between observations. See the
    #     scikit-learn documentation for details on the types of distance metrics
    #     supported. The default is mahalanobis distance, and is the gold standard for
    #     similarity analysis in causal inference studies.

    #     Args:
    #         test_features: Numeric feature matrix for all test group observations
    #             that should be matched to the control eligible observations.

    #         control_features: Numeric feature matrix for all control group
    #             eligible observations.

    #         distance_metric: The type of pairwise distance metric to calculate
    #             between each test target observation and all test_eligible
    #             observations.

    #         **kwds: Optional keyword parameters to ``sklearn.metrics.pairwise_distances``
    #             that will be passed directly to the distance function. If using a
    #             scipy.spatial.distance metric, the parameters are still metric
    #             dependent. See the scipy docs for usage examples.

    #     Returns:
    #         A distance matrix D of shape (n_test_targets, n_control_eligible) such
    #         that then D_{i, j} is the distance between the ith array from
    #         test_targets and the jth array from control_eligible.
    #     """
    #     return pairwise_distances(
    #         test_features,
    #         control_features,
    #         metric=distance_metric,
    #         n_jobs=-1,
    #         **kwds,
    #     )
