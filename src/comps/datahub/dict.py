"""
Transformer package interface implementation for builtin dictionary data
structures.
"""
from typing import Any, Sequence, Union

import pyarrow as pa

from comps.datahub.base import Base


class Dict(Base):
    """
    Transformer interface extension for builtin dictionaries used as target
    data sources.

    Args:

        data: Dictionary to use as data source for retrieving data.

    Attributes:

        sequence_counts: Dictionary of variables mapped to sequence lengths.
        sequences_equal: Boolean indicating whether the sequences mapped to
            each key in the dictionary are of equal length. Used for knowing
            whether or not conversion to a Pandas DataFrame is valid.
    """

    def __init__(self, data: Any) -> None:
        super().__init__(data)
        self._sequence_counts: dict = {}
        self._equences_equal: bool = False
        self._validate(data)

    def _validate(self, data: dict[str, Any]) -> None:
        """
        Validate that all dictionary values are sequences and set
        sequence_counts, sequences_equal, variables, and shape attributes.
        """
        for key, values in data.items():
            if not isinstance(key, str):
                raise ValueError("Dictionary keys must be type str")

            if not isinstance(values, Sequence):
                raise ValueError("Dictionary values must be type Sequence")

            self._sequence_counts[key] = len(values)
            self.variables[key] = (type(values[0]),)

        self._sequences_equal = (
            True if len(set(self._sequence_counts.values())) == 1 else False
        )

        self.set_shape(
            max(self._sequence_counts.values()), len(self._sequence_counts.keys())
        )

    def select(self, variables: Union[str, Sequence[str]]) -> Union[pa.Array, pa.Table]:
        """
        Select data for specific variables from the data class attribute. For
        dictionaries with keys mapped to sequences of equal length, a
        multi-dimensional NumPy array or recarray can be selected using a
        sequence of variable names. If a dictionary is mapped to sequences of
        data that have different lengths, only one variable at a time can be
        selected and if a sequence of variables is input as arguments, only the
        data for the first variable will be returned as a numpy array.

        Args:

            variables: Variable name or sequence of variable names to return
                data records for.

            as_records: Return multi-variable numpy array as a structured array.

        Returns:

            NumPy ndarray or recarray of data records containing the the
            requested variables' data values in the variable order requested.
        """
        variables_is_str = isinstance(variables, str)
        variables_is_sequence = isinstance(variables, Sequence)

        if not variables_is_str and not variables_is_sequence:
            raise ValueError("Variables argument must be string or sequence of strings")

        if variables_is_str or not self._sequences_equal:
            if not self._sequences_equal and variables_is_sequence:
                variables = variables[0]
                self.selections.append([variables])

            selection = pa.array(self.data[variables])

        else:
            selection = {variable: self.data[variable] for variable in variables}
            self.selections.append(list(selection.keys()))
            selection = pa.table(selection)

        return selection
