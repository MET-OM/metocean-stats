# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shlex

# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'metocean-stats'
copyright = '2023, KonstantinChri'
author = 'KonstantinChri' 

# The full version, including alpha/beta/rc tags
release = 'latest'


# -- General configuration ---------------------------------------------------
extensions = [
'sphinx.ext.mathjax',
]

templates_path = ['_templates']
file_insertion_enabled = True
source_suffix = '.rst'
exclude_patterns = []

# The master toctree document.
master_doc = 'index'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

# -- Additional configuration ------------------------------------------------

# Ensure Sphinx can find the data files
html_extra_path = ['files']
