"""
Observation distance algorithms implemented using scikit-learn.
"""
from inspect import getfullargspec
from typing import Any, Optional, TypeAlias, Union

from numpy import cov, number, reshape
from numpy.linalg import inv
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

SklearnPropensityClassifier: TypeAlias = Union[
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
]


SKLEARN_DISTANCE_ALGORITHMS = {
    "propensity": {
        "boosted_tree": "sklearn.ensemble.GradientBoostingClassifier",
        "decision_tree": "sklearn.tree.DecisionTreeClassifier",
        "logistic": "sklearn.linear_model.LogisticRegression",
        "random_forest": "sklearn.ensemble.RandomForestClassifier",
        "neural_network": "sklearn.neural_network.MLPClassifier",
    },
}


class SklearnDistance:
    """
    Class for making propensity score and covariate distance calculations based
    on Pandas DataFrame inputs using the scikit-learn framework.

    All methods that take a ``data`` argument expect the data to be a

    Attributes:
        classifiers: Dictionary of algorithm names mapped to scikit-learn
            classifier classes that can be used for propensity score modeling
            to predict scores used to calculate propensity score distance
            between target and non-target class observations.
    """

    classifiers = {
        "boosted_tree": GradientBoostingClassifier,
        "decision_tree": DecisionTreeClassifier,
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "neural_network": MLPClassifier,
    }

    def model(
        self,
        data: DataFrame,
        algorithm: str,
        sample_weight: Optional[NDArray[number]] = None,
        **kwargs: Any,
    ) -> SklearnPropensityClassifier:
        """
        Fit a classifier algorithm to the data to create a model instance that
        can be used to calculate propensity scores.

        Args:
            data: Data input with where the first column is a binary target
                class indicator and all other columns are the features used to
                fit the classifier algorithm.

            algorithm: Name of the algorithm to fit to the data to create a
                fitted classifier model object.

            sample_weight: Optional array of sample weights for observations. If
                None, then samples are equally weighted.

            **kwargs: Keyword arguments to pass through to the scikit-learn
                classifier algorithm to set hyperparameters prior to fitting.

        Returns:
            Fitted scikit-learn classifier model instance.
        """
        classifier = self.classifiers[algorithm]
        fit_args = getfullargspec(classifier.fit).args

        labels, features = data.iloc[:, 0], data.iloc[:, 1:]

        if "sample_weight" in fit_args:
            fitted_model = classifier(**kwargs).fit(
                features, labels, sample_weight=sample_weight
            )
        else:
            fitted_model = classifier(**kwargs).fit(features, labels)

        return fitted_model

    def propensities(
        self,
        data: DataFrame,
        algorithm: Optional[str] = None,
        model: Optional[SklearnPropensityClassifier] = None,
        **kwargs: Any,
    ) -> dict[str, NDArray[number]]:
        """
        Dictionary of bundled target class probabilities for target and
        non-target observations in data.

        Args:
            data: Data input with where the first column is a binary target
                class indicator and all other columns are the features that
                used to generate propensity scores via the supplied model.

            algorithm: A classifier algorithm to fit a model for so the model
                can be used to make probability (propensity) predictions for
                all observations.

            model: A scikit-learn binary classifier model instance that was
                fit on the same features present in the provided data.

            **kwargs: Keyword arguments to pass through to the classifier model
                algorithm trainer for the configured compute engine, and or an
                optional sample_weight argument for model fitting.

        Returns:
            Dictionary with 'target' and 'non_target' keys mapped to array of
            propensity scores that indicate the modeled probability that an
            observation belongs to the target class. Scores within each array
            are in the order that the target and non-target observations appear
            in the data.
        """
        if not model:
            assert algorithm
            model = self.model(data, algorithm, **kwargs)

        target_probabilities = model.predict_proba(  # type: ignore
            data[data.iloc[:, 0] == 1].iloc[:, 1:]
        )[:, 1]  # type: NDArray[number]

        non_target_probabilities = model.predict_proba(  # type: ignore
            data[data.iloc[:, 0] == 0].iloc[:, 1:]
        )[:, 1]  # type: NDArray[number]

        return {"target": target_probabilities, "non_target": non_target_probabilities}

    def propensity_distances(
        self,
        data: Optional[DataFrame] = None,
        algorithm: Optional[str] = None,
        model: Optional[SklearnPropensityClassifier] = None,
        propensities: Optional[dict[str, NDArray[number]]] = None,
        **kwargs: Any,
    ) -> NDArray[number]:
        """
        Pairwise absolute differences between target and non-target observation
        propensity scores.

        Args:
            data: Data input with where the first column is binary target class
                indicator and all other columns are the features that will be
                used to generate propensity scores using the supplied model.

            algorithm: A classifier algorithm to fit a model for so the model
                can be used to make probability (propensity) predictions for
                all observations.

            model: A scikit-learn binary classifier model instance that was
                fit on the same features present in the provided data.

            propensities: Dictionary with 'target' and 'non_target' keys mapped
                to array of propensity scores.

            **kwargs: Keyword arguments to pass through to the classifier model
                algorithm trainer for the configured compute engine, and or an
                optional sample_weight argument for model fitting.

        Returns:
            Pairwise m x n (target x non-target) ndarray of absolute differences
            between target and non-target observation propensity scores.
        """
        if not propensities:
            assert isinstance(data, DataFrame) and (isinstance(algorithm, str) or model)
            propensities = self.propensities(
                data, algorithm=algorithm, model=model, **kwargs
            )

        target_propensities = reshape(
            propensities["target"],
            (len(propensities["target"]), 1),
        )

        non_target_propensities = reshape(
            propensities["non_target"],
            (1, len(propensities["non_target"])),
        )

        return abs(target_propensities - non_target_propensities)

    def covariate_distances(
        self,
        data: DataFrame,
        algorithm: str,
        **kwargs: Any,
    ) -> NDArray[number]:
        """
        Use a covariate distance algorithm to calculate distances between
        target observations and non-target observations.

        Args:
            data: Data input with where the first column is binary target class
                indicator and all other columns are the features used to
                calculate covariate distance.

            algorithm: A covariate distance metric available for the
                ``sklearn.metrics.pairwise_distances`` function.

            **kwargs: Optional keyword parameters to pass to the scikit-learn
                pairwise_distances function.

        Returns:
            Pairwise m x n (target x non-target) ndarray of distances between
            target and non-target observations.
        """
        target_features = data[data.iloc[:, 0] == 1].iloc[:, 1:]
        non_target_features = data[data.iloc[:, 0] == 0].iloc[:, 1:]

        if algorithm != "mahalanobis":
            distances = pairwise_distances(
                target_features,
                non_target_features,
                metric=algorithm,
                n_jobs=-1,
                **kwargs,
            )

        else:
            covariance = cov(data.iloc[:, 1:].T)
            inverse_covariance = inv(covariance)
            distances = pairwise_distances(
                target_features,
                non_target_features,
                metric=algorithm,
                VI=inverse_covariance,
                n_jobs=-1,
                **kwargs,
            )

        return distances

    def distances(
        self,
        data: DataFrame,
        algorithm: str = "logistic",
        **kwargs: Any,
    ) -> NDArray[number]:
        """
        Calculate the distance between each target observation and all
        non-target observations using the requested algorithm and create a
        ndarray matrix where each target observation i is represented as a row
        and each non-target observation j is represented as a column and the
        value at row i, column j is the distance between target observation i
        and non-target observation j.

        Args:
            data: Data input with where the first column is binary target class
                indicator and all other columns are the features used to fit the
                classifier algorithm.

            algorithm: A valid propensity score distance classifier or
                covariate distance algorithm name.

            **kwargs: Keyword arguments to pass through to the classification
                algorithm class initializer for the configured compute engine
                or to the scikit learn pairwise_distances function if the
                algorithm is a covariate distance metric.

        Returns:
            Pairwise m x n (target x non-target) ndarray of distances between
            target and non-target observations.
        """
        if algorithm in self.classifiers:
            return self.propensity_distances(
                data,
                algorithm,
                **kwargs,
            )

        return self.covariate_distances(
            data,
            algorithm,
            **kwargs,
        )
