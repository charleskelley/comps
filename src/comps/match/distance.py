"""
Module with different methods for calculating different types of covariate
distance between observations.

MatchIt Distance Methods Implemented

|x| = Implemented

Propensity Score

|x| 1. Generalized linear model (glm) - logistic regression
       * pypark, dask, sklearn - LogisticRegression

    2. Generalized additive model (gam)

    3. Generalized boosted model (gbm)
       * pyspark - GBTClassifier (gbt)

|x| 4. Lasso, ridge, or elasticnet (lasso, ridge, elasticnet)

    5. Classification tree (rpart)
       * pyspark - DecisionTreeRegressor

|x| 6. Random Forest classification (randomforest)
    7. Neural Network (nnet) - single-layer hidden network
    8. Covariate balancing propensity scores (cbps)
    9. Bayesian additive regression trees (bart)

Covariate Distances

|x| 1. Euclidean (euclidean)
|x| 2. Scaled euclidean (scaled_euclidian)
|x| 3. Mahalanobis distance (mahalanobis)
    4. Robust Mahalanobis distance (robust_mahalanobis)
"""


class Distance:
    """
    Methods for calculating pairwise distances between target and non-target
    observations using various propensity score and covariate distance methods.
    """

    pass
