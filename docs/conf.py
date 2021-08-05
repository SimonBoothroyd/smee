# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.pardir))


# -- Project information -----------------------------------------------------

project = 'SMIRNOFF Energy Evaluations'
copyright = "2021, Simon Boothroyd"
author = 'Simon Boothroyd'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'nbsphinx_link',
]

source_suffix = '.rst'

master_doc = 'index'

language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# Autodoc settings
autosummary_generate = True

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# Set up the intershinx mappings.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'openff.toolkit': ('https://open-forcefield-toolkit.readthedocs.io/en/latest/', None),
    'openff.interchange': ('https://openff-interchange.readthedocs.io/en/latest/', None),
}

# Set up mathjax.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'prev_next_buttons_location': None,
    'sticky_navigation': False
}

html_static_path = ['_static']

html_context = {
    'css_files': [
        '_static/css/theme_overrides.css',  # override wide tables in RTD theme
    ],
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'smirnoffeedoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'smirnoffee.tex', 'SMIRNOFF-EE Documentation', author, 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'smirnoffee', 'SMIRNOFF-EE Documentation', [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'smirnoffee', 'SMIRNOFF-EE Documentation',
     author, 'smirnoffee', 'Differentiably evaluate energies of molecules using SMIRNOFF force fields.',
     'Miscellaneous'),
]
