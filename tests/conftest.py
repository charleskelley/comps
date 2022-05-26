import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import vaex
from pyspark.sql import DataFrame, SparkSession
from vaex.dataframe import DataFrameLocal


@pytest.fixture
def data_path():
    """Resolved PosixPath to the ``/tests/data`` directory"""
    return Path(__file__).resolve(strict=True).parent.joinpath("data")


class Bunch(dict):
    """
    Container object exposing keys as attributes. Bunch extends dictionaries by
    enabling values to be accessed by key, `bunch["value_key"]`, or by an
    attribute, `bunch.value_key`.

    .. note:

       THIS CODE WAS COPIED FROM ``scikit-learn.utils.Bunch`` FOR TEST
       DEPENDENCY STABILITY.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


@pytest.fixture
def data_dict_factory(data_path):
    """Fixture factory for dictionary representation of CSV test data"""

    def load_dict(*csv_pathsegments: str) -> dict:
        with data_path.joinpath(*csv_pathsegments).open() as openfile:
            dicts_list = list(csv.DictReader(openfile))

        data_dict = {key: [] for key in dicts_list[0].keys()}  # type: dict

        for row in dicts_list:
            for key in data_dict.keys():
                data_dict[key].append(row[key])

        return data_dict

    return load_dict


@pytest.fixture
def data_numpy_factory(data_path):
    """Fixture factory for NumPy structured array representation of CSV test data"""

    def load_structured_array(*csv_pathsegments: str) -> np.ndarray:
        csv_fpath = data_path.joinpath(*csv_pathsegments)
        structured_array = pd.read_csv(csv_fpath).to_records()  # type: ignore

        return structured_array

    return load_structured_array


@pytest.fixture
def data_pandas_factory(data_path):
    """Fixture factory for Pandas DataFrame representation of CSV test data"""

    def load_pandas_dataframe(*csv_pathsegments: str) -> pd.DataFrame:
        csv_fpath = data_path.joinpath(*csv_pathsegments)
        pandas_dataframe = pd.read_csv(csv_fpath)

        return pandas_dataframe

    return load_pandas_dataframe


@pytest.fixture
def data_polars_factory(data_path):
    """Fixture factory for Polars DataFrame representation of Parquet test data"""

    def load_polars_dataframe(*parquet_pathsegments: str) -> pl.DataFrame:
        parquet_fpath = data_path.joinpath(*parquet_pathsegments)
        polars_dataframe = pl.read_parquet(parquet_fpath)

        return polars_dataframe

    return load_polars_dataframe


@pytest.fixture
def data_pyarrow_factory(data_path):
    """Fixture factory for PyArrow Table representation of Parquet test data"""

    def load_pyarrow_table(*parquet_pathsegments: str) -> pa.Table:
        parquet_fpath = str(data_path.joinpath(*parquet_pathsegments))
        pyarrow_table = pa.parquet.read_table(parquet_fpath)

        return pyarrow_table

    return load_pyarrow_table


@pytest.fixture
def data_pyspark_factory(data_path):
    """Fixture factory for PySpark DataFrame representation of Parquet test data"""

    def load_pyspark_dataframe(*parquet_pathsegments: str) -> DataFrame:
        spark = SparkSession.builder.appName(
            "data_pyspark_factory-".format(str(time.time()))
        ).getOrCreate()
        parquet_fpath = str(data_path.joinpath(*parquet_pathsegments))
        pyspark_dataframe = spark.read.parquet(parquet_fpath)

        return pyspark_dataframe

    return load_pyspark_dataframe


@pytest.fixture
def data_vaex_factory(data_path):
    """Fixture factory for Vaex Table representation of Parquet test data"""

    def load_vaex_dataframe(*parquet_pathsegments: str) -> DataFrameLocal:
        parquet_fpath = str(data_path.joinpath(*parquet_pathsegments))
        vaex_dataframe = vaex.open(parquet_fpath)

        return vaex_dataframe

    return load_vaex_dataframe


@pytest.fixture
def data_bunch(
    data_dict_factory,
    data_numpy_factory,
    data_pandas_factory,
    data_polars_factory,
    data_pyarrow_factory,
    data_pyspark_factory,
    data_vaex_factory,
):
    """
    Container object with short version bank datasets in all the key data
    structure types that the transormer (xfmr) subpackage supports.

    Each dataset type can be accessed via the following key/attribute:

    * dict - builtin dict
    * numpy - structured array
    * pandas - DataFrame
    * polars - DataFrame
    * pyarrow - table
    * pyspark - DataFrame
    * vaex - DataFrame
    """

    def load_data_bunch(csv_pathsegments: tuple, parquet_pathsegments: tuple) -> Bunch:
        data_dict = data_dict_factory(*csv_pathsegments)
        data_numpy = data_numpy_factory(*csv_pathsegments)
        data_pandas = data_pandas_factory(*csv_pathsegments)
        data_polars = data_polars_factory(*parquet_pathsegments)
        data_pyarrow = data_pyarrow_factory(*parquet_pathsegments)
        data_pyspark = data_pyspark_factory(*parquet_pathsegments)
        data_vaex = data_vaex_factory(*parquet_pathsegments)

        bunch = Bunch(
            bdict=data_dict,
            numpy=data_numpy,
            pandas=data_pandas,
            polars=data_polars,
            pyarrow=data_pyarrow,
            pyspark=data_pyspark,
            vaex=data_vaex,
        )

        return bunch

    return load_data_bunch


@pytest.fixture
def pd_bank_data():
    """Returns bank data in pandas DataFrame for testing.

    Returns
    -------
    pandas : DataFrame
        Pandas DataFrame with bank data.
    """
    pass


@pytest.fixture
def int_array():
    """Returns 10x4 NumPy array for testing.

    Returns
    -------
    numpy : Array
        10x4 array with int 0 to 9 for each value.
    """
    return np.array(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [1, 3, 5, 7],
            [7, 5, 3, 1],
            [2, 4, 6, 8],
            [0, 1, 2, 3],
            [8, 6, 4, 2],
            [1, 3, 5, 7],
            [3, 5, 7, 9],
            [0, 1, 2, 3],
        ],
        np.int64,
    )
