#######
Roadmap
#######

.. topic:: Comps package core objective

   Provide an opinionated package to support fundamental techniques,
   frameworks, models, and visualizations used for comparative and causal
   analysis

======
Vision
======

Comparison of different sets or subsets of data in search of effect causation
is the foundation of analytics across business and research settings.
Unfortunately, using Python to implement advanced comparative and causal
analysis techniques requires the accumulation of specialized packages and
writing setup code to run and summarize each analysis.

The Comps package provides a single opinionated resource to support the
application of various comparison techniques with minimal data prep and
transformation. To achieve this, the key principles guiding development are:

.. code-block:: text

   `Pareto Principle`_ (80/20 rule) prioritization

   Simplicity over flexibility

   Opinionated analytics methodology

   Excellent documentation

   Lazy computation wherever possible

.. _Pareto Principle: https://en.wikipedia.org/wiki/Pareto_principle

=================
Development Focus
=================

------------------------
Evaluating Distributions
------------------------

The first step of comparison is to evaluate whether there are in fact
differences between groups that may need to be adjusted for. If two groups are
already randomly distributed across all key traits that influence behavior in
the target domain of consideration, then observation matching or group
recomposition may not be necessary.

* Summarization of key metrics using measures of central tendency and dispersion
* Quantifying overlap of metric distributions between groups or segments
* Visualization of overlap of metric distributions between groups or segments
* Standardized report for comparison between groups and documentation of proper
  interpretation to guide next steps

----------------
Matching Methods
----------------

* Select matched observations between groups to support direct comparison
* Select subsets of observations to maximize similarity at the group level

Planned order for matching method implementation.

Stratification:

1. Exact matching
2. Coarsened exact matching

Modeling:

1. Classic distance modeling
   * Nearest neighbor greedy or optimal
     1. Mahalanobis distance
     2. Propensity score estimated with logit
2. Machine learning
   * D-AEMR
   * Propensity score estimated with random forest and cross-validation
   * Matching frontier
   * Genetic matching

-------------------
Statistical Testing
-------------------

* Perform testing between groups for directional and incremental differences
* Clearly document requirements that must be met for different types of tests
  to support proper use and interpretation

1. Two sample T-testing
2. Average treatment effect of the treated (ATT)
3. Time series break/trend analysis

------------
RFM analysis
------------

* Support Recency Frequency Monetary (RFM) for group comparison

---------------------------
Reporting and visualization
---------------------------

* Provide reporting and visualization methods for testing results
* Develop Preso package for more elaborate reporting that can be used generally
