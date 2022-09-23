"""
Standardization of access and tracking of data from various data management data
structures used in common Python data management packages like Pandas, Polars,
PyArrow, PySpark, Vaex, etc...
"""
from comps.xfmr.base import available_data_packages, data_structure_type
from comps.xfmr.dict import Dict
from comps.xfmr.pandas import Pandas


__all__ = [
    "available_data_packages",
    "data_structure_type",
    "Dict",
    "Pandas",
]
