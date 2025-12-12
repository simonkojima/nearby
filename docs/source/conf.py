# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import nearby

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Nearby"
copyright = "2025, Nearby contributors"
author = "Simon Kojima, Alex Pepi, Fabien Lotte"
release = nearby.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_multiversion",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

smv_tag_whitelist = r"^v\d+\.\d+.*$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {"image_light": "logo-light.png", "image_dark": "logo-dark.png"},
    "navbar_end": ["theme-switcher", "version-switcher"],
    "switcher": {
        "json_url": "https://simonkojima.github.io/nearby-docs/versions.json",
        "version_match": release,
    },
}
html_context = {
    "version_match": release,
}

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"example_",
}
