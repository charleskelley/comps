import numpy as np
import pytest

from comps.datahub.pandas import Pandas

pystestmark = pytest.mark.unit


@pytest.fixture
def bank_pandas(data_pandas_factory):
    """Bank test data a dictionary of variable: sequence key-values"""
    bank = data_pandas_factory(*("bank", "bank.csv"))

    return bank


def test_pandas_init(bank_pandas, bank_attributes_set_get):
    """Pandas class instantiation and initialization"""
    bank = Pandas(bank_pandas)

    assert isinstance(bank, Pandas)

    # Test for Base class attributes
    assert bank.data.equals(bank_pandas)
    assert bank.structure == "pandas.core.frame.DataFrame"
    assert bank.size > 0

    # Test proper inheritance of Metadata dataclass
    assert bank_attributes_set_get(bank)


def test_pandas_select(bank_pandas):
    """Selecting variables from Pandas DataFrame and returning NumPy arrays"""
    bank = Pandas(bank_pandas)
    age_sum = bank_pandas.age.sum()
    age_mean = bank_pandas.age.mean()

    single_variable = bank.select("age")
    assert isinstance(single_variable, np.ndarray)
    assert np.sum(single_variable) == age_sum
    assert np.mean(single_variable) == age_mean

    multiple_variables = bank.select(["age", "job", "marital", "balance"])
    assert isinstance(multiple_variables, np.ndarray)
    assert np.sum(multiple_variables[:, 0]) == age_sum
    assert np.mean(multiple_variables[:, 0]) == age_mean

    multiple_variables = bank.select(
        ["age", "job", "marital", "balance"], as_records=True
    )
    assert isinstance(multiple_variables, np.recarray)
    assert np.sum(multiple_variables.age) == age_sum
    assert np.mean(multiple_variables.age) == age_mean
