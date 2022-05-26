###########
Development
###########

====================
Comps Package Design
====================

The Comps package is designed around using NumPy arrays as the foundational
data structure for data processing. This design decision was made for two
reasons.

1. Maintain separation of data storage and in-memory data wrangling from
   computation and processing needs--thereby maximizing upstream data interface
   flexibility and minimizing downstream dependencies
2. Leverage the tightly-coupled performance oriented NumPy computational
   ecosystem including libraries like SciPy, scikit-learn, Numba, etc...

=============================
Transformer Subpackage Design
=============================

The transformer (xfmr) subpackage creates a common interface that can
interact with various types of Python data structures in a consistent manner.
This allows users of the Comps package to easily pass different types of
complex and simple data structures to Comps functions and methods and without
custom preprocessing.

For example, one can pass a Pandas DataFrame or a Spark DataFrame to a Comps
function using the same arguments, and the underlying data will be properly
processed based on DataFrame type completely without additional user input.

Complex data packages/structures currently supported are:

* ``pandas`` DataFrame
* ``polars`` DataFrame
* ``pyarrow`` or Apache Arrow table
* ``pyspark`` or Apache Spark DataFrame
* ``vaex-core`` local DataFrame

In addition to the above complex data structures, support for basic data
structures is also available with more limited functionality.

Basic data structures current supported are:

* Builtin dictionaries
* ``numpy`` structured arrays

===================
Apple Silicon Notes
===================

* SciPy requires version >= 1.8 for OpenBLAS compatibility
