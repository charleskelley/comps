from typing import Any, Sequence, Union

from comps.datahub.base import Base


class Pandas(Base):
    """
    Transformer interface extension for using Pandas DataFrame as a data
    source.

    Args:
        data: Pandas DataFrame to use as data source for retrieving data.

    Attributes:
    """

    def __init__(self, data: Any) -> None:
        super().__init__(data)
        self._validate(data)

    def _validate(self, data: Any) -> None:
        """
        Validate that target data structure is a Pandas DataFrame and
        initialize inherited transformer Base class attributes.
        """
        self.set_shape(*self.data.shape)

        for name, dtype in zip(
            list(self.data.columns), [x.name for x in self.data.dtypes]
        ):
            self.variables[name] = (dtype,)

    def select(self, variables: Union[str, Sequence[str]], as_records: bool = False):
        if isinstance(variables, str):
            array = self.data[variables].to_numpy()

        else:
            array = self.data[list(variables)]
            array = array.to_numpy() if not as_records else array.to_records()

        return array

    def calculate(self):
        pass
