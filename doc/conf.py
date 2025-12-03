# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

# pylint: skip-file

import os
import os.path as osp
import sys

import guidata.config as gcfg
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from guidata.utils import qt_scraper


class OptionsTableDirective(Directive):
    """Custom directive to include dynamically generated options table."""

    has_content = False

    def run(self):
        """Generate and include the options table."""
        from sigima.config import options

        # Get the RST content
        rst_content = options.generate_rst_doc()

        # Create a container node
        container = nodes.container()

        # Parse the RST content and add it to the container
        rst_lines = rst_content.splitlines()
        string_list = StringList(rst_lines)
        self.state.nested_parse(string_list, self.content_offset, container)

        return [container]


sys.path.insert(0, os.path.abspath(".."))

import sigima

# Turn off validation of guidata config
# (documentation build is not the right place for validation)
gcfg.set_validation_mode(gcfg.ValidationMode.DISABLED)


def exclude_api_from_gettext(app):
    """Exclude detailed API docs from gettext extraction.

    This excludes API docs but keeps api/index.rst for translation.
    """
    if app.builder.name == "gettext":
        # Get all RST files in the api directory
        api_dir = osp.join(app.srcdir, "api")
        if osp.exists(api_dir):
            for filename in os.listdir(api_dir):
                if filename.endswith(".rst") and filename != "index.rst":
                    # Remove .rst extension and add wildcard
                    pattern = f"api/{filename[:-4]}*"
                    if pattern not in app.config.exclude_patterns:
                        app.config.exclude_patterns.append(pattern)

                # Also check subdirectories (may be useful in the future)
                for dirname in os.listdir(api_dir):
                    subdir_path = osp.join(api_dir, dirname)
                    if osp.isdir(subdir_path):
                        # Exclude entire subdirectories except their index files
                        pattern = f"api/{dirname}/*"
                        if pattern not in app.config.exclude_patterns:
                            app.config.exclude_patterns.append(pattern)

        # Suppress warnings about excluded API documents during gettext builds
        app.config.suppress_warnings.extend(["toc.excluded", "ref.doc"])


def patch_datalab_client_example():
    """Patch the datalab_client example to use stub server during doc build.

    This function modifies the example execution context so that when
    Sphinx-Gallery runs datalab_client.py, it connects to a stub server
    instead of requiring a real DataLab instance.
    """
    from sigima.client.stub import patch_simpleremoteproxy_for_stub

    # Start stub server and apply the patch
    return patch_simpleremoteproxy_for_stub()


def setup(app):
    """Setup function for Sphinx."""
    app.add_directive("options-table", OptionsTableDirective)
    app.connect("builder-inited", exclude_api_from_gettext)


# -- Project information -----------------------------------------------------

project = "Sigima"
author = ""
copyright = "2025, DataLab Platform Developers"
release = sigima.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "guidata.dataset.autodoc",
    "sphinx_gallery.gen_gallery",
]
templates_path = ["_templates"]
exclude_patterns = [
    "sg_execution_times.rst",
    "**/sg_execution_times.rst",
]

# Suppress the warning about unpicklable sphinx_gallery_conf
# (it contains reset_modules function which cannot be pickled)
suppress_warnings = ["config.cache"]

# -- Sphinx-Gallery configuration --------------------------------------------
# Using guidata's generic Qt scraper for capturing all Qt widgets
# Configure to use the last widget as thumbnail for a complete pipeline view
qt_scraper.set_qt_scraper_config(
    thumbnail_source="last", hide_toolbars=True, capture_inside_layout=True
)
sphinx_gallery_conf = qt_scraper.get_sphinx_gallery_conf(
    filename_pattern="", min_reported_time=60, show_memory=False
)

if "READTHEDOCS" in os.environ or "CI" in os.environ:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Patch datalab_client example to use stub server during documentation build
_stub_server = None


def _reset_example_namespace(gallery_conf, fname):
    """Reset namespace and setup stub server for datalab_client example."""
    global _stub_server
    # Cleanup previous stub server if it exists
    if _stub_server is not None:
        _stub_server.stop()
        _stub_server = None
    # Setup new stub server for datalab_client example
    if "datalab_client" in fname:
        _stub_server = patch_datalab_client_example()


# Add reset handler to sphinx_gallery_conf
sphinx_gallery_conf["reset_modules"] = _reset_example_namespace
sphinx_gallery_conf["reset_modules_order"] = "before"
sphinx_gallery_conf["subsection_order"] = [
    "./examples/getting_started",
    "./examples/use_cases",
    "./examples/features",
]
sphinx_gallery_conf["within_subsection_order"] = "ExampleTitleSortKey"
# Note: The handler also cleans up the stub server from previous examples

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = project
html_logo = "images/Sigima-Banner.svg"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False
templates_path = ["_templates"]
# if "language=fr" in sys.argv:
#     ann = ""  # noqa: E501
# else:
#     ann = ""  # noqa: E501
html_theme_options = {
    "show_toc_level": 2,
    "github_url": "https://github.com/DataLab-Platform/Sigima/",
    "logo": {
        "text": f"v{sigima.__version__}",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/sigima",
            "icon": "_static/pypi.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "CODRA",
            "url": "https://codra.net",
            "icon": "_static/codra.png",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "DataLab",
            "url": "https://datalab-platform.com",
            "icon": "_static/DataLab.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    # "announcement": ann,
}
html_static_path = ["_static"]

# -- Options for LaTeX output ------------------------------------------------
latex_logo = "_static/Sigima-Frontpage.png"

# -- Options for sphinx-intl package -----------------------------------------
locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False
gettext_location = False

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
    "guidata": ("https://guidata.readthedocs.io/en/latest/", None),
}
