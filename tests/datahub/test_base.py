import pytest

from comps.datahub.base import (
    Base,
    Metadata,
    available_data_packages,
    data_structure_type,
)


pystestmark = pytest.mark.unit


PACKAGES_STRUCTURES = {
    "numpy": "numpy.recarray",
    "pyarrow": "pyarrow.lib.Table",
    "pandas": "pandas.core.frame.DataFrame",
    "pyspark": "pyspark.sql.dataframe.DataFrame",
}


def test_available_data_packages():
    """Discovery of data structure packages or modules in runtime environment"""
    supported_packages = set(PACKAGES_STRUCTURES.keys())

    assert available_data_packages() == supported_packages


@pytest.mark.parametrize(
    "bunch_attr",
    ["bdict"] + list(PACKAGES_STRUCTURES.keys()),
)
def test_data_structure_type(bank_bunch, bunch_attr):
    """Identification of data structure's data type"""
    if bunch_attr == "bdict":
        assert str(data_structure_type(bank_bunch[bunch_attr])) == "dict"
    else:
        assert (
            str(data_structure_type(bank_bunch[bunch_attr]))
            == PACKAGES_STRUCTURES[bunch_attr]
        )


@pytest.fixture
def metadata():
    """Empty Metadata class object instance"""
    metadata_instance = Metadata()

    return metadata_instance


def test_metadata_class_init(metadata):
    """Metadata class instantiation and initialization"""
    assert isinstance(metadata, Metadata)


def test_metadata_class_attributes(metadata, bank_metadata, bank_attributes_set_get):
    """Metadata class attribute value setting and getting"""
    attributes = list(bank_metadata.keys())

    assert len(metadata.__dict__.keys()) == len(attributes)
    assert bank_attributes_set_get(metadata)


def test_base_class_init(data_dict_factory, bank_attributes_set_get):
    """Base class instantiation and initialization"""
    bank_bdict = data_dict_factory(*("bank", "bank.csv"))

    base = Base(bank_bdict)

    assert isinstance(base, Base)
    assert base.data == bank_bdict
    assert base.structure == "dict"
    assert base.size > 0

    # Test proper inheritance of Metadata dataclass
    assert bank_attributes_set_get(base)
