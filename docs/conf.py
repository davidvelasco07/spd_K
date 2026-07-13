"""Sphinx configuration for spd_K."""

project = "spd_K"
author = "spd_K developers"
release = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
]

source_suffix = [".rst", ".md"]
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath", "amsmath"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "spd_K"
html_theme_options = {
    "source_repository": "https://github.com/davidvelasco07/spd_K",
    "source_branch": "main",
    "source_directory": "docs/",
}

# GitHub Pages project site: https://<user>.github.io/<repo>/
html_baseurl = "https://davidvelasco07.github.io/spd_K/"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
