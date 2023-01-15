"""
Key functions and classes used to define the standardized transformer interface.
"""
import logging
from dataclasses import dataclass, field
from importlib import import_module
from sys import getsizeof
from typing import Any, NamedTuple, Optional, Sequence, Set, Union

from numpy import recarray
from numpy.typing import NDArray

# Package names mapped to tuples with mddule iddentifiers and attributes as two
# item tuples for easy dynamic importing
DATA_PACKAGE_STRUCTURES = {
    "numpy": ("numpy", "recarray"),
    "pyarrow": ("pyarrow.lib", "Table"),
    "pandas": ("pandas.core.frame", "DataFrame"),
    "pyspark": ("pyspark.sql.dataframe", "DataFrame"),
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

    for package in DATA_PACKAGE_STRUCTURES.keys():
        try:
            import_module(package)
            packages.add(package)
        except ModuleNotFoundError:
            logging.info("{0} package not available".format(package))

    return packages


def data_structure_type(data: Any, package_only: bool = False) -> str:
    """
    Identify the particular type of data structure. Specifically, whether the
    object passed to the function is a Pandas, Polars, Spark, or Vaex
    DataFrame, an Arrow table, or a NumPy structured array.

    Args:

        data: An instance of a core data structure from one of the
            valid data packages.

        package_only: Whether to return only the package name for the data
            structure rather than full path to the type within package.

    Returns:

         String representation of the package module import path of class or
         type of data structure or simply the data management package name. See
         the enumeration list below for the valid import paths.


         - numpy.recarray
         - pandas.core.frame.DataFrame
         - polars.internals.frame.DataFrame
         - pyarrow.lib.Table
         - pyspark.sql.dataframe.DataFrame
         - vaex.dataframe.DataFrameLocal
    """
    if isinstance(data, dict):
        return "dict"

    data_structures = {
        package: structure for package, structure in DATA_PACKAGE_STRUCTURES.items()
    }

    for package, structure in data_structures.items():
        module = import_module(structure[0])
        structure_type = getattr(module, structure[1])

        if isinstance(data, structure_type):
            return ".".join(structure) if not package_only else package

    raise ValueError("Invalid 'data_structure' argument provided")


# NamedTuple for shape attribute in Metadata class
Shape = NamedTuple("Shape", [("records", int), ("variables", int)])

# NamedTuple for variables attribute in Metadata class
Variable = NamedTuple("Variable", [("data_type", str), ("description", str)])


@dataclass
class Metadata:
    """
    Dataclass for holding standardized data structure metadata. The class is
    designed for downstream package development interface consistency and for
    larger applications where metadata tracking for documentation and
    presentation is highly beneficial.

    In most ad hoc use cases of transformer objects, interacting directly with
    the native metadata attributes of the data structure instance held in the
    'data' attribute of the 'Base' class is more efficient.

    Attributes:

        description: Overview of the data within the data structure or of
            the source and nature of the data structure.

        extras: Dictionary of extra metadata attributes and values not covered
            by other attributes.

        name: Identifying name of the data structure.

        record_keys: Dictionary of different types of record keys mapped to
            variable name lists that identify the variables that make up that
            key type where the possible key types are.

                - unique: Name or names of variables that  make each individual
                  record unique from other records in the data structure.
                - entity: The identifying entity that an individual record is
                  mapped to or belongs to. For example a record belonging to a
                  customer may have a 'cust_id' variable as the record key
                  where each record belongs to one and only one customer.
                - temporal: The primary time grain of the record. For example,
                  events may be captured with a timestamp for each event where
                  as dimensions or aggregates may be captured as of the end of
                  a day and marked with an effective date.

        shape: Named tuple 'Shape' with attributes 'records', and 'variables'
            representing the maximum number of records in the data structure
            followed by the number of variables. For tabular data structures
            the number of records is equal to the row count and the variable
            number is equal to the number of columns. For key-value data
            structures, the number of variables is equal to the number of keys,
            and the number of records is equal to the longest sequence of
            values mapped to a key.

        size: Estimated size in bytes of the data structure in memory.

        variables: Dictionary metadata for all valid variables that can be part
            of an individual record in the data structure. The structure of the
            metadata holding dictionary is shown below where each variable name
            is a key in the dictionary mapped to a named tuple 'Variable' with
            two attributes 'description' and 'data_type' providing the
            variable's description or definition and data type respectively::

                {
                    'variable_name': Variable(
                        data_type='variable_data_type'
                        description='variable_description',
                    )
                }


            For tabular data structures, variable names are the column names and
            for dictionaries the variable names are the keys of the dictionary.
    """

    description: str = field(default_factory=str)
    extras: dict[str, Any] = field(default_factory=dict)
    name: str = field(default_factory=str)
    record_keys: dict[str, list[str]] = field(default_factory=dict)
    shape: NamedTuple = Shape(0, 0)
    size: int = field(default_factory=int)
    variables: dict[str, Any] = field(default_factory=dict)

    def set_shape(self, records: int, variables: int) -> None:
        """
        Set the class shape attribute as namedtuple Shape with attributes
        variables and records. Default is records=0 and variables=0, which
        this method will overwrite.
        """
        self.shape = Shape(records, variables)

    def set_variable(
        self, variable: str, data_type: str, description: str = ""
    ) -> None:
        """
        Set a variable in the class variables attribute as namedtuple Variable
        with attributes data_type and description. Default is data_type='' and
        description='', which this method will overwrite using argument values.
        """
        self.variables[variable] = Variable(
            data_type=data_type, description=description
        )


class Base(Metadata):
    """
    The transformer class provides a standardized interface managing data using
    various types of data structures, particularly DataFrames, to support the
    creation make domain specific relationships between the different data
    structure instances or elements.

    Args:

        data: The data structure object instance that the Base class template
            defines a standardized data interface for.

    Attributes:

        aggregates: Dictionary of lists of group by variables mapped to aggregate
            structured arrays for those variables. These are cached results
            from the use of the aggregate method with cache equal to True.

        compute: The type of compute scheme (local or dask) to use when
            creating when interacting with the transformer object.

        data: Instance of a data structure where data can be accessed. The data
            structure's data should be accessed using Xfmr class methods, but
            one can also access the data directly using methods native to the
            particular data structure such as Pandas DataFrame slicing or PySpark
            DataFrame methods.

        metadata: Dictionary with miscellaneous metadata for the data structure,
            provided by the user or inherited from the wrapped data structure
            type if available.

        selections: List of lists of string variable names for selections in appended
            to list in each time the selection method is used.

        structure: String name or dot separated module path of the data
            structure instance assigned to the data attribute.

        size: In memory size of the data structure assigned to the data
            attribute calculated using ``sys.getsizeof``.
    """

    def __init__(self, data: Any) -> None:
        super().__init__()
        self.aggregates: Optional[dict[Union[str, Set[str]], NDArray[Any]]] = None
        self.compute: str = "local"
        self.data: Any = data
        self.selections: list[list[str]] = []
        self.structure: Any = data_structure_type(data)
        self.size = getsizeof(self.data)

    def select(
        self,
        variables: Union[str, Sequence[str]],
        where: Optional[str] = None,
        as_records: bool = False,
    ) -> Union[NDArray, recarray]:
        """
        Select data for specific variables from the data structure assigned to
        the data class attribute.

        Args:

            variables: Variable name or sequence of variable names to return
                data for.

            where: String expression that evaluates to a boolean using Python
                evaluate. This option uses the Pandas DataFrame.query framework
                where you can refer to column names that are not valid Python
                variable names by surrounding them in backticks. Thus, column
                names containing spaces or punctuations (besides underscores)
                or starting with digits must be surrounded by backticks.

            as_records: Return selected data as NumPy structured or record
                array rather than basic multi-dimensional array.

        Returns:

            NumPy multi-dimensional or structured array of data records
            containing the the requested variables' data values.
        """
        raise NotImplementedError

    def calculate(
        self,
        calcs: dict[str, Union[str, Sequence[str]]],
        by: Union[str, Sequence[str]],
        cache: bool = False,
    ) -> dict[Union[str, list, dict], NDArray]:
        """
        Create an aggregate series from a data structure by grouping data by the
        provided set of variables and applying an aggregation operator to
        specified variables.

        Args:

            calcs: Calculation or aggregation operator name mapped to a
                variable name or names to apply operator to within the data
                structure, using the window specified by the group by variable
                names.

            by: Name or names of variables to group the aggregate
                operator calculations by.

            cache: Whether to cache the aggregation in the 'aggregates'
                attribute in the Xfmr class instance.

        Returns:

            NumPy structured array with group by variables and aggregate
            variables arrays.
        """
        raise NotImplementedError
