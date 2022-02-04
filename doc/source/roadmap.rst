Roadmap
#######

.. topic:: Comps package core objective

   Provide an opinionated package to support fundamental techniques,
   frameworks, models, and visualizations used for advanced data comparison

Vision 
======

Comparison of different sets or subsets of data in search of effect causation
is the foundation of analytics across business and research settings.
Unfortunately, using Python to implement advance comparison and causal analysis
techniques requires the accumulation of many specialized packages and writing
setup code to run and summarize each analysis.

The Comps package provides a single resource to support the application of
various comparison techniques with minimal data prep and transformation. To
achieve this, the key principles guiding development are:

* `Pareto Principle`_ (80/20 rule) prioritization
* Simplicity over flexibility
* Opinionated configuration
* Lazy computation
* Good documenation

.. _Pareto Principle: https://en.wikipedia.org/wiki/Pareto_principle

Development focus
=================

Matching methods
----------------

* Select matched observations between groups to support direct comparison
* Select subsets of observations to maximize similarity at the group level

Statistical testing
-------------------

* Perform testing between groups for directional and incremental differences 

RFM analysis
------------
* Support Recency Frequency Monetary (RFM) for group comparison 

Reporting and visualization
---------------------------

* Provide reporting and visualization methods for testing results 

