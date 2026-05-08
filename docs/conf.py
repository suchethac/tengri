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
    "sphinx_gallery.gen_gallery",
]

# -- Sphinx-Gallery ----------------------------------------------------------

# Pin matplotlib savefig DPI so gallery thumbnails render at the same
# resolution across local builds and CI. Set before sphinx-gallery's first
# figure save. (sphinx-gallery dropped its top-level ``savefig_dpi`` key on
# Sphinx 9 / sphinx-gallery 0.18; rcParams is the supported path.)
try:
    import matplotlib

    matplotlib.rcParams["savefig.dpi"] = 150
except Exception:  # pragma: no cover - matplotlib missing on docs-only env
    pass

# ── Skip-execution list, derived from what is already pre-rendered ─────────
# Policy: do NOT re-execute any script whose main figure
# ``docs/auto_examples/<group>/images/sphx_glr_<name>_001.png`` is already
# committed. If the image is missing, the script will execute on the next
# ``make html`` and produce one. So the contract is "fetch the current ones,
# only reproduce if missing" — driven by what's on disk, not a hand-curated
# list. Heavy inference scripts stay fast because their pre-rendered output
# is committed; lightweight new scripts auto-rebuild on first build.
import pathlib as _pl

_AUTO_EXAMPLES = _pl.Path(__file__).resolve().parent / "auto_examples"
_DO_NOT_EXECUTE = sorted(
    {
        p.stem.replace("sphx_glr_", "").rsplit("_", 1)[0]
        for p in _AUTO_EXAMPLES.glob("*/images/sphx_glr_plot_*_001.png")
    }
)
# Regex-safe alternation; falls back to a never-match sentinel so the
# negative-lookahead below still matches every plot_*.py.
_skip_alt = "|".join(_DO_NOT_EXECUTE) or "__never_match_anything__"

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    # Run any plot_*.py whose basename is NOT in _DO_NOT_EXECUTE.
    "filename_pattern": rf"^(?!.*({_skip_alt}))plot_.+\.py$",
    # ignore_pattern HIDES files from the gallery entirely. Reserved for
    # truly-broken WIP scripts; everything else stays visible via the
    # precompute path above.
    "ignore_pattern": r"^$",
    "download_all_examples": False,
    # Locally we execute (default). On CI (e.g. GitHub Actions sets CI=true) we
    # use the pre-rendered docs/auto_examples/ that the developer committed so
    # the cloud build doesn't need SSP grids, RELAGN, or ~20 GB of optional
    # data. Run `make html` locally and commit the regenerated auto_examples/
    # before pushing if you want fresh galleries.
    "plot_gallery": "False" if os.environ.get("CI", "").lower() == "true" else "True",
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
    "thumbnail_size": (320, 224),
    "default_thumb_file": None,
    "matplotlib_animations": False,
    "abort_on_example_error": False,
    "min_reported_time": 2,
}


# ── Gallery-index post-processor ────────────────────────────────────────────
# sphinx-gallery writes `auto_examples/index.rst` with one ``.. toctree::`` at
# the very bottom, *after* the last category heading ("X-ray Emission").  In
# RST, content following a heading belongs to that heading, so every other
# gallery subsection ends up nested under X-ray in the sidebar.
#
# Fix: move the toctree to the *top* of the file (right after the ":orphan:"
# directive) so it is not a child of any section.  We run this after
# sphinx-gallery has generated its index.
def _fix_gallery_index_toctree(app, *_args, **_kwargs):
    from pathlib import Path

    path = Path(app.srcdir) / "auto_examples" / "index.rst"
    if not path.exists():
        return
    src = path.read_text()
    # Capture the toctree block appended at the bottom.
    import re

    tc_re = re.compile(
        r"\n\.\. toctree::\s*\n(?:\s+:\w+:.*\n)*(?:\s+.*\n)*",
        re.MULTILINE,
    )
    matches = list(tc_re.finditer(src))
    if not matches:
        return
    tc_block = matches[-1].group(0).strip("\n")
    # Strip the original position.
    src_no_tc = src[: matches[-1].start()] + src[matches[-1].end() :]
    # Insert after ``:orphan:`` or at the very top.
    if ":orphan:" in src_no_tc:
        new = src_no_tc.replace(
            ":orphan:",
            ":orphan:\n\n" + tc_block,
            1,
        )
    else:
        new = tc_block + "\n\n" + src_no_tc
    path.write_text(new)


def setup(app):
    # Priority 1000 runs *after* sphinx-gallery's own builder-inited handler.
    app.connect("builder-inited", _fix_gallery_index_toctree, priority=1000)

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3  # auto-generate IDs for h1–h3 (lets [text](#anchor) work)

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- nbsphinx (Jupyter → HTML via nbconvert) ---------------------------------

nbsphinx_execute = "never"

# -- Theme configuration -----------------------------------------------------

html_theme = "furo"
html_baseurl = "https://suchethac.github.io/tengri/"
html_title = "tengri"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_logo = "_static/tengri-logo.png"
html_favicon = "_static/tengri-logo.png"

html_theme_options = {
    "source_repository": "https://github.com/suchethac/tengri",
    "source_branch": "main",
    "source_directory": "docs/",
    # Use a single logo (transparent PNG) for both light and dark modes —
    # Furo otherwise renders two images side-by-side.
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
    "internal",
    "**.ipynb_checkpoints",
    "superpowers",
    "specs",
    # Sphinx-gallery internal outputs that shouldn't be picked up as source
    "auto_examples/index.rst.new",
    "sg_execution_times.rst",
    "auto_examples/**/*.py",
    "auto_examples/**/*.md5",
    "auto_examples/**/*.codeobj.json",
    "auto_examples/**/*.zip",
    "auto_examples/**/*.ipynb",
    # Narrative sections superseded by repo root notebooks/ spine
    "getting_started/**",
    "forward_model/**",
    "inference/**",
    "fitting/**",
    "advanced/**",
    "developer/**",
    "dev/**",
    "install.md",
    "known_bugs.md",
    "NEBULAR_REFACTOR.md",
    "_notebooks/**",
    # Not part of the published sidebar (content folded into index.md or omitted)
    "changelog.md",
    "documentation.md",
]
