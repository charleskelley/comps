import csv
import time
from ast import literal_eval
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import DataFrame, SparkSession


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
def data_dict_factory(data_path) -> object:
    """Fixture factory for dictionary representation of CSV test data"""

    def load_dict(*csv_pathsegments: str) -> dict:
        with data_path.joinpath(*csv_pathsegments).open() as openfile:
            dicts_list = list(csv.DictReader(openfile))

        data = {key: [] for key in dicts_list[0].keys()}  # type: dict

        for row in dicts_list:
            for key in data.keys():
                try:
                    value = literal_eval(row[key])
                except (SyntaxError, ValueError):
                    value = row[key]

                data[key].append(value)

        return data

    return load_dict


@pytest.fixture
def data_numpy_factory(data_path) -> object:
    """Fixture factory for NumPy structured array representation of CSV test data"""

    def load_structured_array(*csv_pathsegments: str) -> np.recarray:
        csv_fpath = data_path.joinpath(*csv_pathsegments)
        data = pd.read_csv(csv_fpath)  # type: ignore
        data = data.convert_dtypes().to_records(index=False)

        return data

    return load_structured_array


@pytest.fixture
def data_pandas_factory(data_path):
    """Fixture factory for Pandas DataFrame representation of CSV test data"""

    def load_pandas_dataframe(*csv_pathsegments: str) -> pd.DataFrame:
        csv_fpath = data_path.joinpath(*csv_pathsegments)
        data = pd.read_csv(csv_fpath)
        data = data.convert_dtypes()

        return data

    return load_pandas_dataframe


@pytest.fixture
def data_pyarrow_factory(data_path):
    """Fixture factory for PyArrow Table representation of Parquet test data"""

    def load_pyarrow_table(*parquet_pathsegments: str) -> pa.Table:
        parquet_fpath = str(data_path.joinpath(*parquet_pathsegments))
        data = pq.read_table(parquet_fpath)

        return data

    return load_pyarrow_table


@pytest.fixture
def data_pyspark_factory(data_path):
    """Fixture factory for PySpark DataFrame representation of Parquet test data"""

    def load_pyspark_dataframe(*parquet_pathsegments: str) -> DataFrame:
        spark = SparkSession.builder.appName(
            "data_pyspark_factory-{0}".format(str(time.time()))
        ).getOrCreate()
        parquet_fpath = str(data_path.joinpath(*parquet_pathsegments))
        data = spark.read.parquet(parquet_fpath)

        return data

    return load_pyspark_dataframe


@pytest.fixture
def data_bunch(
    data_dict_factory,
    data_numpy_factory,
    data_pandas_factory,
    data_pyarrow_factory,
    data_pyspark_factory,
):
    """
    Container object with short version bank datasets in all the key data
    structure types that the transormer (datahub) subpackage supports.

    Each dataset type can be accessed via the following key/attribute:

    * dict - builtin dict
    * numpy - structured array
    * pandas - DataFrame
    * pyarrow - table
    * pyspark - DataFrame
    """

    def load_data_bunch(csv_pathsegments: tuple, parquet_pathsegments: tuple) -> Bunch:
        data_dict = data_dict_factory(*csv_pathsegments)
        data_numpy = data_numpy_factory(*csv_pathsegments)
        data_pandas = data_pandas_factory(*csv_pathsegments)
        data_pyarrow = data_pyarrow_factory(*parquet_pathsegments)
        data_pyspark = data_pyspark_factory(*parquet_pathsegments)

        bunch = Bunch(
            bdict=data_dict,
            numpy=data_numpy,
            pandas=data_pandas,
            pyarrow=data_pyarrow,
            pyspark=data_pyspark,
        )

        return bunch

    return load_data_bunch


@pytest.fixture
def bank_bunch(data_bunch):
    """Data bunch for /tests/data/bank datasets"""
    bunch = data_bunch(("bank", "bank.csv"), ("bank", "bank.parquet"))

    return bunch


@pytest.fixture
def lalonde_variables():
    variables = {
        "features": [
            "age",
            "education",
            "black",
            "hispanic",
            "married",
            "nodegree",
            "re75",
        ],
        "outcome": "re78",
        "target": "treatment",
    }

    return variables


@pytest.fixture
def lelonde_bunch(data_bunch):
    """Data bunch for /tests/data/lelonde datasets"""
    bunch = data_bunch(("lelonde", "lelonde.csv"), ("lelonde", "lelonde.parquet"))

    return bunch


@pytest.fixture
def data_factory(
    data_dict_factory,
    data_numpy_factory,
    data_pandas_factory,
    data_pyarrow_factory,
    data_pyspark_factory,
):
    """
    Dataset factory to return specific dataset as a specific data structure
    type where data_struct_type is one of <bdict|numpy|pandas|pyarrow|pyspark>.
    """

    def load_dataset(
        dataset_name: str, data_struct_type: str
    ) -> Union[DataFrame, pd.DataFrame, pa.Table, np.recarray, dict]:
        """Load dataset using structure factory based on data_struct_type argument"""
        dataset_pathsegments = {
            "bank_csv": ("bank", "bank.csv"),
            "bank_parquet": ("bank", "bank.parquet"),
            "lalonde_csv": ("lalonde", "lalonde.csv"),
            "lalonde_parquet": ("lalonde", "lalonde.parquet"),
        }
        pathsegments_key = (
            f"{dataset_name}_parquet"
            if data_struct_type in {"pyarrow", "pyspark"}
            else f"{dataset_name}_csv"
        )
        pathsegments = dataset_pathsegments[pathsegments_key]

        data_struct_factory = {
            "dict": data_dict_factory,
            "numpy": data_numpy_factory,
            "pandas": data_pandas_factory,
            "pyarrow": data_pyarrow_factory,
            "pyspark": data_pyspark_factory,
        }

        return data_struct_factory[data_struct_type](*pathsegments)

    return load_dataset


@pytest.fixture
def bank_metadata():
    """Metadata for bank test dataset"""
    metadata = {
        "description": (
            "The data is related with direct marketing campaigns of a Portuguese"
            " banking institution. The marketing campaigns were based on phone"
            " calls. Often, more than one contact to the same client was required,"
            " in order to assess whether the bank's term deposit product was"
            " subscribed to.",
        ),
        "extras": {
            "source": (
                "UCI Machine Learning Repository"
                " <https://archive-beta.ics.uci.edu/ml/datasets/bank+marketing>"
            ),
            "published_year": 2011,
            "citation": (
                "[Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data"
                " Mining for Bank Direct Marketing: An Application of the CRISP-DM"
                " Methodology."
            ),
        },
        "name": "bank_mktg_cmpgn",
        "record_keys": {"unique": ["recordid"], "entity": ["recordid"], "temporal": []},
        "shape": (4521, 18),
        "size": 0,
        "variables": {
            "age": {
                "data_type": "numeric",
                "description": "Contact's age in years",
            },
            "job": {
                "data_type": "string",
                "description": (
                    "Contact's type of job (admin, unknown, unemployed, management,"
                    " housemaid, entrepreneur, student, blue-collar, self-employed,"
                    " retired, technician, services)"
                ),
            },
            "marital": {
                "data_type": "string",
                "description": (
                    "Contact's marital status (married, divorced, single) Note that"
                    " divorced means divorced or widowed"
                ),
            },
            "education": {
                "data_type": "string",
                "description": (
                    "Contacts highest level of education (unknown, secondary,"
                    " primary, tertiary)"
                ),
            },
            "default": {
                "data_type": "string",
                "description": "Whether contact has credit in default? (yes,no)",
            },
            "balance": {
                "data_type": "numeric",
                "description": "Contact's average yearly balance, in euros",
            },
            "housing": {
                "data_type": "string",
                "description": "Whether contact has housing loan? (yes,no)",
            },
            "loan": {
                "data_type": "string",
                "description": "Whether contact has personal loan? (yes,no)",
            },
            "contact": {
                "data_type": "string",
                "description": (
                    "Communication type of contact (unknown, telephone, cellular)"
                ),
            },
            "day": {
                "data_type": "numeric",
                "description": "Day of month of last contact",
            },
            "month": {
                "data_type": "string",
                "description": (
                    "Month of year of last contact (jan, feb, mar, ..., nov, dec)"
                ),
            },
            "duration": {
                "data_type": "numeric",
                "description": "Duration of last contact in seconds",
            },
            "campaign": {
                "data_type": "numeric",
                "description": (
                    "Number of contacts performed during this campaign and for this"
                    " client (includes last contact)"
                ),
            },
            "pdays": {
                "data_type": "numeric",
                "description": (
                    "Number of days that passed by after the client was last contacted"
                    " from a previous campaign (numeric, -1 means client was not"
                    " previously contacted)"
                ),
            },
            "previous": {
                "data_type": "numeric",
                "description": (
                    "Number of contacts performed before this campaign and for this"
                    " client (numeric)"
                ),
            },
            "poutcome": {
                "data_type": "string",
                "description": (
                    "Outcome of the previous marketing campaign (unknown, other,"
                    " failure, success)"
                ),
            },
            "y": {
                "data_type": "string",
                "description": (
                    "Target outcome has the client subscribed a term deposit in the"
                    " campaign? (yes, no)"
                ),
            },
            "recordid": {
                "data_type": "numeric",
                "description": "Unique ID for individual prospect within dataset",
            },
        },
    }

    return metadata


@pytest.fixture
def bank_attributes_set_get(bank_metadata):
    """T/F can set and retrieve all metadata attributes on object"""

    def set_get_attributes(class_object: object):
        attributes = list(bank_metadata.keys())

        for attribute in attributes:
            assert hasattr(class_object, attribute)
            if attribute != "shape":
                setattr(class_object, attribute, bank_metadata[attribute])
            else:
                class_object.set_shape(*bank_metadata[attribute])

        attributes_match_list = []

        for attribute in attributes:
            attributes_match_list.append(
                getattr(class_object, attribute) == bank_metadata[attribute]
            )

        return all(attributes_match_list)

    return set_get_attributes


@pytest.fixture
def int_matrix_10x4():
    """10x4 NumPy ndarray for testing"""
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
        np.int8,
    )


@pytest.fixture
def float_array_10():
    """10 item float array for testing"""
    return np.array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        np.float16,
    )


@pytest.fixture
def float_matrix_10x1(float_array_10):
    """10x1 float matrix for testing"""
    return np.reshape(float_array_10, (10, 1))


@pytest.fixture
def float_matrix_1x10(float_array_10):
    """1x10 float matrix for testing"""
    return np.reshape(float_array_10, (1, 10))
