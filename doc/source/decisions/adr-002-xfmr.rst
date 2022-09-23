####################################
ADR-001 Transformer(xfmr) Subpackage
####################################

In order to make interaction with many different types of input data structures
easy, the transformer or ``comps.xfmr`` subpackage was created to provide a
standardized interface to common analytics and data science data management
data structures such as Pandas, PySpark, and Vaex DataFrames.

:Status: Accepted
:Created: 2022-05-15
:Last Updated: 2022-06-05

=======
Context
=======

Within the Python data analytics and data science community there are quite a
few popular packages used for managing data, each of which provide slightly
different APIs and data management methods. Each package is useful in different
situations but the differing APIs make quickly applying standardized analytics
methods difficult because the data must be extracted and transformed into the
proper format before being used for computation.

It would be a very nice if Comps package users could pass different types of
data structures to functions and methods using a standardized call pattern
without having to first extract the data from the target data structure.

For example, passing a Pandas, Polars, or Vaex DataFrame to a Comps function or
method should be the same process and result in the same output given the
underlying data is the same::

  Pandas:  data_summary = comps.summarize(data=pandas_dataframe, plots=True)

  Polars:  data_summary = comps.summarize(data=polars_dataframe, plots=True)

  Vaex:    data_summary = comps.summarize(data=vaex_dataframe, plots=True)

========
Decision
========

Create a class that can act as a wrapper to different types of data structures
and provide a standardized interface for accessing data within those data
structures via a singular set of methods and attributes.

============
Consequences
============

The transformer subpackage provides an invaluable layer of extraction for
designing a top level API for the Comps package. For example, low level
functions and methods of the Comps package can be designed and implemented
using NumPy data structures while a high level API can be crafted that accepts
multiple types of data structure inputs. The transformer class object can act
as an interpretation and transformation layer that creates proper NumPy inputs
for low level functions from multiple types of non-NumPy data structures, all
unbeknownst to the user.

.. note::

   This is the template in [Documenting architecture decisions - Michael
   Nygard](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions).
