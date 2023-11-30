# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sphinx_rtd_theme
import sys

sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'daart'
copyright = '2023, matt whiteway'
author = 'matt whiteway'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # allows automatic parsing of docstrings
    'sphinx.ext.mathjax',  # allows mathjax in documentation
    'sphinx.ext.viewcode',  # links documentation to source code
    'sphinx.ext.githubpages',  # allows integration with github
    'sphinx.ext.napoleon',  # parsing of different docstring styles
    'sphinx_automodapi.automodapi',
    'sphinx_copybutton',  # add copy button to code blocks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']


# document constructors
def skip(app, what, name, obj, skip, options):
    if name == '__init__':
        return False
    return skip


def setup(app):
    app.connect('autodoc-skip-member', skip)
