"""
Module of different matching methods used to match target observations to
non-target observations based on feature similarity.

MatchIt Matching Methods

1. Cardinality (cardinality) matching and other forms of matching - using
   mixed integer programming using GLPK library, rather than forming pairs,
   cardinality matching selects the largest subset of units that satisfies
   user-supplied balance constraints on mean differences.

2. Coarsened exact matching (cem) - where covariates are coarsened into bins,
   and a complete cross of the coarsened covariates is used to form subclasses
   defined by each combination of the coarsened covariate levels. Any subclass
   that doesn’t contain both treated and control units is discarded, leaving
   only subclasses containing treatment and control units that are exactly
   equal on the coarsened covariates.

3. Exact matching (exact) - a complete cross of the covariates is used to
   form subclasses defined by each combination of the covariate levels. Any
   subclass that doesn’t contain both treated and control units is discarded,
   leaving only subclasses containing treatment and control units that are
   exactly equal on the included covariates.

4. Optimal full matching (full) - a form of subclassification wherein all
   units, both treatment and control (i.e., the "full" sample), are assigned to
   a subclass and receive at least one match. The matching is optimal in the
   sense that that sum of the absolute distances between the treated and
   control units in each subclass is as small as possible

5. Genetic matching (genetic) - a form of nearest neighbor matching where
   distances are computed as the generalized Mahalanobis distance, which is a
   generalization of the Mahalanobis distance with a scaling factor for each
   covariate that represents the importance of that covariate to the
   distance.

6. Nearest neighbor (nearest) - performs greedy nearest neighbor matching. A
   distance is computed between each treated unit and each control unit, and,
   one by one, each treated unit is assigned a control unit as a match. The
   matching is "greedy" in the sense that there is no action taken to optimize
   an overall criterion; each match is selected without considering the other
   matches that may occur subsequently.

7. Optimal matching (optimal) - performs optimal pair matching. The matching is
   optimal in the sense that that sum of the absolute pairwise distances in the
   matched sample is as small as possible.

8. Subclass Matching (subclass) - performs subclassification on the distance
   measure (i.e., propensity score). Treatment and control units are placed
   into subclasses based on quantiles of the propensity score in the treated
   group, in the control group, or overall, depending on the desired estimand.
   Weights are computed based on the proportion of treated units in each
   subclass.
"""
