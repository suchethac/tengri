"""Sphinx configuration for tengri documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "tengri"
author = "Suchetha Cooray"
release = "0.1.0"
copyright = "2026, Suchetha Cooray"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
    "tasklist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- nbsphinx configuration --------------------------------------------------

nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# -- Theme configuration -----------------------------------------------------

html_theme = "furo"
html_baseurl = "https://suchethac.github.io/tengri/"
html_title = "tengri"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "source_repository": "https://github.com/suchethac/tengri",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True

# -- napoleon (numpydoc-style) -----------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Exclude patterns --------------------------------------------------------

exclude_patterns = [
    "_build",
    "_notebooks",
    "internal",
    "**.ipynb_checkpoints",
    "superpowers",
    "specs",
]
