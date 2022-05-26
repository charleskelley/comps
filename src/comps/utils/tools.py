"""
Utility functions helpful for general usage
"""
import csv
import os
import re
from pathlib import Path
from typing import Any, Iterable, Union

import pkg_resources
import yaml


def pkg_path() -> Path:
    """Returns absolute root path for package as Path object"""
    return Path(pkg_resources.resource_filename(__name__, "")).parent


# def user_path() -> Path:
#     """Path to user's home directory on Windows, Unix, or macOS"""
#     platform, user = sys.platform, getuser()

#     if bool(re.match(r"win.*", platform)):
#         drive = os.path.splitdrive(sys.executable)[0]
#         user_path = Path(drive).joinpath("Users", user).resolve(strict=True)
#     else:
#         home = os.getenv("HOME")
#         user_path = Path(home).resolve(strict=True) if home else Path("~").expanduser()

#     return user_path

# def resolved_path(*pathsegments: Union[str, os.PathLike]) -> Path:
#     """Strictly resolved path from pathsegments"""
#     return Path(*pathsegments).resolve(strict=True)


# def text_str(text_fpath: Union[str, os.PathLike]) -> Union[str, list[str]]:
#     """Returns single string representation of text file"""
#     with open(resolved_path(text_fpath).resolve(strict=True)) as openfile:
#         text = openfile.read()

#     return text


def csv_2col_dict(csv_fpath: os.PathLike[str]) -> dict[str, str]:
    """Returns a key: value dictionary from two column CSV"""
    with open(Path(csv_fpath).resolve(strict=True)) as openfile:
        csv_dict = dict(filter(None, csv.reader(openfile)))

    return csv_dict


def csv_dicts_list(csv_fpath: os.PathLike[str]) -> list[dict[str, str]]:
    """
    Returns a list of dictionaries where header CSV row is keys for each dict
    and values in each dict are corresponding column value of rows > 1
    """
    with open(Path(csv_fpath).resolve(strict=True)) as openfile:
        dicts_list = list(csv.DictReader(openfile))

    return dicts_list


def yaml_dict(yaml_fpath: os.PathLike[str]) -> dict[str, Union[Any, dict[str, Any]]]:
    """Returns a dictionary with imported YAML file data"""
    file = Path(yaml_fpath).resolve(strict=True)
    with open(file) as openfile:
        yaml_dict = yaml.safe_load(openfile)

    return yaml_dict  # type: ignore[no-any-return]


def list_filter_match(
    iterable: Iterable[str], repattern: Union[str, re.Pattern[str]]
) -> list[str]:
    """Take an iterable and filters with regex match and returns a list"""
    pattern = re.compile(repattern)

    return list(filter(lambda x: bool(pattern.match(x)), iterable))
