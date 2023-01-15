"""
Standalone test module for the ``data_bunch`` fixture to ensure accuracy
because the fixture encapsalates all the the different data structure testing
fixtures used throughout Comps and Transformer (datahub) subpackage tests.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

# Sum of 'age' column in smaller bank dataset
SUM_BANK_AGE = 186130


def test_data_dict_factory(data_dict_factory):
    """Fixture factory for loading of CSV data into dictionary"""
    bank_dict = data_dict_factory("bank", "bank.csv")

    assert type(bank_dict) == dict
    assert sum(list(map(int, bank_dict["age"]))) == SUM_BANK_AGE


def test_data_numpy_factory(data_numpy_factory):
    """Fixture factory for loading of CSV data into numpy structured array"""
    bank_structured_array = data_numpy_factory("bank", "bank.csv")

    assert type(bank_structured_array) == np.recarray
    assert np.sum(bank_structured_array.age) == SUM_BANK_AGE


def test_data_pandas_factory(data_pandas_factory):
    """Fixture factory for loading of CSV data into Pandas DataFrame"""
    bank_pandas_dataframe = data_pandas_factory("bank", "bank.csv")

    assert type(bank_pandas_dataframe) == pd.DataFrame
    assert bank_pandas_dataframe.age.sum() == SUM_BANK_AGE


def test_data_pyarrow_factory(data_pyarrow_factory):
    """Fixture factory for loading of Parquet data into PyArrow DataFrame"""
    bank_pyarrow_table = data_pyarrow_factory("bank", "bank.parquet")

    assert type(bank_pyarrow_table) == pa.Table
    assert pa.compute.sum(bank_pyarrow_table["age"]).as_py() == SUM_BANK_AGE


def test_data_pyspark_factory(data_pyspark_factory):
    """Fixture factory for loading of Parquet data into PySpark DataFrame"""
    bank_pyspark_dataframe = data_pyspark_factory("bank", "bank.parquet")

    assert type(bank_pyspark_dataframe) == DataFrame
    assert bank_pyspark_dataframe.agg(F.sum("age")).head()[0] == SUM_BANK_AGE


def test_data_bunch(bank_bunch):
    """Ensure ``data_bunch`` returns equivalent data for all data structure attributes
    """
    # bank_bunch = data_bunch(("bank", "bank.csv"), ("bank", "bank.parquet"))

    assert type(bank_bunch.bdict) == dict
    assert sum(list(map(int, bank_bunch.bdict["age"]))) == SUM_BANK_AGE

    assert type(bank_bunch.numpy) == np.recarray
    assert np.sum(bank_bunch.numpy.age) == SUM_BANK_AGE

    assert type(bank_bunch.pandas) == pd.DataFrame
    assert bank_bunch.pandas.age.sum() == SUM_BANK_AGE

    assert type(bank_bunch.pyarrow) == pa.Table
    assert pa.compute.sum(bank_bunch.pyarrow["age"]).as_py() == SUM_BANK_AGE

    assert type(bank_bunch.pyspark) == DataFrame
    assert bank_bunch.pyspark.agg(F.sum("age")).head()[0] == SUM_BANK_AGE
