"""
Module for transformation and tracking of data in various data management data
structures used in common Python data management packages like Pandas, Polars,
PyArrow, PySpark, Vaex, etc...
"""
import logging
from importlib import import_module
from typing import Any, Set

from numpy import ndarray


# Package names mapped to tuples with mddule iddentifiers and attributes as two
# item tuples for easy dynamic importing
DATA_PACKAGES_STRUCTURES = {
    "numpy": ("numpy", "recarray"),
    "pyarrow": ("pyarrow.lib", "Table"),
    "pandas": ("pandas.core.frame", "DataFrame"),
    "polars": ("polars.internals.frame", "DataFrame"),
    "pyspark": ("pyspark.sql.dataframe", "DataFrame"),
    "vaex": ("vaex.dataframe", "DataFrameLocal"),
}


def available_data_packages() -> Set[str]:
    """
    Set or subset of the data management packages installed with data structure
    support available where the complete set tested for includes ``pyarrow``
    (arrow), ``pandas``, ``polars``, ``pyspark`` (spark), ``vaex``, and
    ``numpy`` (only for structured or record arrays).

    Returns:
        Set including one or more string package names (numpy, pyarrow, pandas,
        polars, pyspark, and vaex).
    """
    packages = set()

    for package in DATA_PACKAGES_STRUCTURES.keys():
        try:
            import_module(package)
            packages.add(package)
        except ModuleNotFoundError:
            logging.info("{0} package not available".format(package))

    return packages


def data_structure_type(data_structure: Any, package_only: bool = False) -> str:
    """
    Identify the particular type of data structure. Specifically, whether the
    object passed to the function is a Pandas, Polars, Spark, or Vaex
    DataFrame, an Arrow table, or a numpy structured array.

    Args:
        data_structure: An instance of a core data structure from one of the
            valid data packages.
        package_only: Whether to return only package name rather than full path
            to type within package.

    Returns:
        Package class or type of data structure. One of pyarrow.lib.Table,
        pandas.core.frame.DataFrame, polars.internals.frame.DataFrame,
        pyspark.sql.dataframe.DataFrame, vaex.dataframe.DataFrameLocal,
        numpy.recarray.
    """
    if isinstance(data_structure, dict):

        return "dict"

    if isinstance(data_structure, ndarray):
        if not data_structure.dtype.names:

            return (
                ".".join(DATA_PACKAGES_STRUCTURES["numpy"])
                if not package_only
                else "numpy"
            )

    data_structures = {
        package: structure
        for package, structure in DATA_PACKAGES_STRUCTURES.items()
        if package != "numpy"
    }

    for package, structure in data_structures.items():
        module = import_module(structure[0])
        structure_type = getattr(module, structure[1])

        if isinstance(data_structure, structure_type):

            return ".".join(structure) if not package_only else package

    raise ValueError("Invalid 'data_structure' argument provided")


class Xfmr:
    """
    The transformer class provides a standardized interface for storing and
    managing data using various types of data structures, but particularly
    DataFrames, to support the creation of make domain specific relationships
    between the different data structure intances or elements.

    Additionally, it provides a metadata library that can be used to add
    additional metadata about the data structures and relationships encoded
    into the container elements.

    Args:
        frame: PyArrow table or array of PyArrow tables stored in the table
            attribute by name and can accessed via slicing.
        stack: xfmr.Stack class instance or list of instances.

    Attributes:
        packages: Set of names of valid data managment packages currently
            available in the runtime environment.
        arrays: Dict of pyarrow.array class instance or list of instances.
        tables: PyArrow tables are stored in the table attribute by name and
            can accessed via slicing.
        deques: dict of collections.deque class instances.
        names: Names of all the data structures in the transformer.
        mdata: xfmr.Metadata class instance
    """

    def __init__(self) -> None:
        pass
