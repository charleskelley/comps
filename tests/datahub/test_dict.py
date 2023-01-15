from statistics import mean

import numpy as np
import pyarrow as pa
import pytest

from comps.datahub.dict import Dict

pystestmark = pytest.mark.unit


@pytest.fixture
def bank_dict(data_dict_factory):
    """Bank test data a dictionary of variable: sequence key-values"""
    bank = data_dict_factory(*("bank", "bank.csv"))

    return bank


def test_pyarrow_array(bank_dict):
    """Conversion of dictionary values to PyArrow array with best data type"""
    pass


def test_dict_class_init(bank_dict, bank_attributes_set_get):
    """Bdict class instantiation and initialization"""
    bank = Dict(bank_dict)

    assert isinstance(bank, Dict)

    # Test for Base class attributes
    assert bank.data == bank_dict
    assert bank.structure == "dict"
    assert bank.size > 0

    # Test proper inheritance of Metadata dataclass
    assert bank_attributes_set_get(bank)

    # Test for Bdict initialization attributes
    assert np.mean(list(bank._sequence_counts.values())) == 4521
    assert bank._sequences_equal


def test_dict_select(bank_dict):
    """Selecting variables from dictionary and returning NumPy arrays"""
    bank = Dict(bank_dict)
    age_sum = sum([int(x) for x in bank_dict["age"]])
    age_mean = mean([int(x) for x in bank_dict["age"]])

    single_variable = bank.select("age")
    assert isinstance(single_variable, pa.Array)
    assert pa.compute.sum(single_variable).as_py() == age_sum
    assert pa.compute.mean(single_variable).as_py() == age_mean

    multiple_variables = bank.select(["age", "job", "marital", "balance"])
    assert isinstance(multiple_variables, pa.Table)
    assert pa.compute.sum(multiple_variables["age"]).as_py() == age_sum
    assert pa.compute.mean(multiple_variables["age"]).as_py() == age_mean
