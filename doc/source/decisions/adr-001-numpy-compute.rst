############################
ADR-001 NumPy Compute Engine
############################

To ensure the Comps package is performant and can use the largest set of
established tools for initial development, NumPy ndarrays and NumPy based
computational frameworks (i.e. scikit-learn) were chosen.

:Status: Accepted
:Created: 2022-05-01
:Last Updated: 2022-06-05

=======
Context
=======

The NumPy computational framework is the default for number crunching in Python
but is somewhat limiting when moving from in-memory data single machine
computation to out-of-memory distributed computation required for big data
development. The question then, is whether to develop the the Comps package
using NumPy as the core computational framework when it will likely introduce
many issues when working with big data?

========
Decision
========

To start out, the NumPy ndarray data structure and computational framework will
be used while keeping in mind distruted compute should target using the
[Ray framework](https://www.ray.io) wherever possible and if it is easy to
implement something using the Ray framework from the start, it should be done.

============
Consequences
============

Using NumPy as the core for the Comps package will likely introduce issues when
working with big data in the enterprise and research settings. However, 90% of
use cases should be covered using large ephemeral cloud VM instances where
needed and there are simply too many pros to using the NumPy ecosystem for the
Comps package's advance computational requirements for test control matching
and for common ML and statistics libraries like scikit-learn, SciPy, and
statsmodels.

.. note::

   This is the template in [Documenting architecture decisions - Michael
   Nygard](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions).
