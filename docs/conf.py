# conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('..')) # Add project root to path

# -- Project information -----------------------------------------------------
project = 'DQNChess'
copyright = '2025, IliasSoultana'
author = 'IliasSoultana'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Automatically generate docs from docstrings
    'sphinx.ext.napoleon',     # Support for Google-style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = []
