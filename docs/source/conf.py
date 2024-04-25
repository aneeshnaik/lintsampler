import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'lintsampler'
copyright = '2024, Aneesh Naik'
author = 'Aneesh Naik'
extensions = ['sphinx.ext.autodoc', 'numpydoc', 'myst_nb']
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'sphinx_book_theme'
html_theme_options = {"use_sidenotes": True}
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
myst_heading_anchors = 2