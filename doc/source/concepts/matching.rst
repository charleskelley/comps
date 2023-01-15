################
Matching Methods
################

Matching as it is known today is one of several statistical techniques that
emerged in the 1980s with the aim of estimating causal effects [1]_. Matching
can be defined as any method that “strategically subsamples” a dataset [2]_,
with the aim of balancing observable covariate distributions in the treated and
control groups such that both groups share an equal probability of treatment.

Alternatives, or complements, to matching include: “adjusting for background
variables in a regression model; instrumental variables; structural equation
modeling or selection models.” [3]_

An important distinction from other statistical approaches is that matching is
only the first step in a two-step process to estimate the treatment effect. It
prepares the data for further statistical analysis, but it is not a stand-alone
estimation technique in and of itself. (In machine learning terms, it is part
of the data preprocessing step not the modeling step.) Matching is followed by
difference-in-average estimation (assuming sufficient covariate coverage),
linear regression, or any other modeling method. Proponents of this technique
proclaim that one of the greatest conveniences to using matching is precisely
this flexibility: preprocessing with matching can be followed by “whatever
statistical method you would have used without matching.” [4]_

==============
Exact Matching
==============

With exact matching, a complete cross of the covariates is used to form
subclasses defined by each combination of the covariate levels. Any subclass
that doesn’t contain both treated and control units is discarded, leaving only
subclasses containing treatment and control units that are exactly equal on the
included covariates.

**Algorithms**

* Z algorithm
* Boyer-Moore
* Knuth-Morris-Pratt
* Aho-Corasick

DataFrame join on.

========================
Coarsened Exact Matching
========================

With coarsened exact matching, covariates are coarsened into bins, and a
complete cross of the coarsened covariates is used to form subclasses defined
by each combination of the coarsened covariate levels. Any subclass that
doesn’t contain both treated and control units is discarded, leaving only
subclasses containing treatment and control units that are exactly equal on the
coarsened covariates. The coarsening process can be controlled by an algorithm
or by manually specifying cutpoints and groupings.

==========================
Subclassification Matching
==========================

Performs subclassification on the distance measure (i.e., propensity score).
Treatment and control units are placed into subclasses based on quantiles of
the propensity score in the treated group, in the control group, or overall,
depending on the desired estimand. Weights are computed based on the proportion
of treated units in each subclass.

====================
Cardinality Matching
====================

Cardinality matching uses mixed integer programming and rather than forming
pairs, cardinality matching selects the largest subset of units that satisfies
user-supplied balance constraints on mean differences.

=========================
Nearest Neighbor Matching
=========================

Greedy nearest neighbor matching, a distance is computed between each treated
unit and each control unit, and, one by one, each treated unit is assigned a
control unit as a match. The matching is "greedy" in the sense that there is
no action taken to optimize an overall criterion; each match is selected
without considering the other matches that may occur subsequently.

================
Genetic Matching
================

Genetic matching is a form of nearest neighbor matching where distances are
computed as the generalized Mahalanobis distance, which is a generalization of
the Mahalanobis distance with a scaling factor for each covariate that
represents the importance of that covariate to the distance.

================
Optimal Matching
================

Performs optimal pair matching where the matching is optimal in the sense that
that sum of the absolute pairwise distances in the matched sample is as small
as possible. The method functionally relies on optmatch::fullmatch().

=====================
Optimal Full Matching
=====================

Full matching is a form of subclassification wherein all units, both treatment
and control (i.e., the "full" sample), are assigned to a subclass and receive
at least one match. The matching is optimal in the sense that that sum of the
absolute distances between the treated and control units in each subclass is as
small as possible.

==============
Quick Matching
==============

Performs generalized full matching, which is a form of subclassification
wherein all units, both treatment and control (i.e., the "full" sample), are
assigned to a subclass and receive at least one match. It uses an algorithm
that is extremely fast compared to optimal full matching, which is why it is
labelled as "quick", at the expense of true optimality. The method is described
in Sävje, Higgins, & Sekhon (2021). The method relies on and is a wrapper for
quickmatch::quickmatch()


==========
References
==========

.. [1] Rosenbaum, P., & Rubin, D. (1983). The Central Role of the Propensity
   Score in Observational Studies for Causal Effects. Biometrika, Vol.70(1),
   41-55. *Note: This is not the first instance of a matching method used in
   published statistical research, but Rubin’s proposal for matching on
   propensity scores propelled the technique into greater esteem in social
   science research.*

.. [2] Morgan, S. & Winship, C. (2015). Counterfactuals and Causal Inference:
   Methods and principles for social research. Cambridge: Cambridge University
   Press. Page 142.

.. [3] Stuart, 2010. Page 2.

.. [4] King, G. (2018). Matching Methods for Causal Inference. Published
   Presentation given at Microsoft Research, Cambridge, MA on 1/19/2018.
   https://gking.harvard.edu/presentations/matching-methods-causal-inference-3.
   Slide 5.
