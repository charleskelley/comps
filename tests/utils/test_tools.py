from pathlib import Path

import pytest

from comps.utils.tools import pkg_path


def test_pkg_path():
    """Package root path"""
    pkg_root_path = (
        Path(__file__).resolve().parent.parent.parent.joinpath("src", "comps")
    )

    assert pkg_path() == pkg_root_path
