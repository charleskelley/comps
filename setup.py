#! /usr/bin/env python
from setuptools import find_packages

DESCRIPTION = "Similarity analysis and matching to support causal analysis and performance comparison"
LONG_DESCRIPTION = """\
Comps brings together many machine learning and statistical methods for
finding look alike entities to ease the selection and management of comparison
groups in observational studies and experimental design to support causal
analysis. It also provides a number of standardized aggregations that are
helpful for comparing different types of metrics between groups.

Comps provides:
- Systematic API for applying and matching methods and assessing the similarity
  of different groups of entities based on common characteristics
- Methods for easily managing groups of entities across time to maintain the
  similarity between groups 
  
All Comps package wheels distributed on PyPI are BSD 3-clause licensed.
"""

DISTNAME = "comps"
MAINTAINER = "Charles Kelley"
MAINTAINER_EMAIL = "cksisu@gmail.com"
URL = "https://comps.readthedocs.io.org"
LICENSE = "BSD (3-clause)"
DOWNLOAD_URL = "https://github.com/cksisu/comps"
VERSION = "0.0.1.dev0"
PYTHON_REQUIRES = ">=3.7"

INSTALL_REQUIRES = ["scikit-learn>=0.21.0", "statsmodels>=0.10.0"]

EXTRAS_REQUIRE = {"all": ["seaborn>=0.10.0", "plotly>=2.0.0"]}

CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9"]


if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 7):
        raise RuntimeError("preso requires python >= {0}.".format(PYTHON_REQUIRES))

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        package_dir={"": "comps"},
        packages=find_packages(where="comps"),
        classifiers=CLASSIFIERS)

