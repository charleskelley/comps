# Configuration file for the Sphinx documentation builder.

# For a full list of configuration options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import re
from datetime import datetime


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# import os
# import sys
# sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "comps"
copyright = f"2021-{datetime.now().year}"

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
# import comps
# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
# version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', comps.__version__)
# version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
# The full version, including alpha/beta/rc tags.
# release = comps.__version__
# print("%s %s" % (version, release))

# The full version, including alpha/beta/rc tags
release = "0.0.1.dev0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# Will change to `root_doc` in Sphinx 4
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This pattern also
# affects html_static_path and html_extra_path.
exclude_patterns = []

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'


# -- Options for HTML output -------------------------------------------------

# Theme to use for HTML and HTML Help pages
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_logo = "_static/comps-logo-black.svg"

html_theme_options = {
    "logo_link": "index",
    "github_url": "https://github.com/cksisu/comps",
    "collapse_navigation": True,
}
