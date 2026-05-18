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
    # Run any plot_*.py whose basename is NOT in _DO_NOT_EXECUTE. Sphinx-
    # gallery applies this with ``re.search`` against the *full source path*
    # (e.g. ``/Users/.../examples/<group>/plot_foo.py``), so ``^plot_`` would
    # never match. Anchor at start of string with ``^``, run the negative-
    # lookahead against the whole path to exclude any basename already in
    # ``_DO_NOT_EXECUTE``, then ``.*plot_<...>.py$`` to pin the filename.
    "filename_pattern": rf"^(?!.*(?:{_skip_alt})).*plot_[^/]+\.py$",
    # ignore_pattern HIDES files from the gallery entirely. Used for heavy
    # NUTS/SVI scripts whose runtime + memory footprint OOMs the build (each
    # NUTS warmup can peak at 20+ GB per CLAUDE.md gotcha). These scripts
    # still run as standalone demos for advanced users.
    "ignore_pattern": (
        # Heavy NUTS/SVI scripts whose runtime + memory footprint OOMs the
        # build (each NUTS warmup can peak at 20+ GB per CLAUDE.md gotcha).
        r"plot_(population_scaling|hierarchical_convergence|prior_posterior_compare)\.py$"
    ),
    "download_all_examples": False,
    # Locally we execute (default). On CI (e.g. GitHub Actions sets CI=true) we
    # use the pre-rendered docs/auto_examples/ that the developer committed so
    # the cloud build doesn't need SSP grids, RELAGN, or ~20 GB of optional
    # data. Run `make html` locally and commit the regenerated auto_examples/
    # before pushing if you want fresh galleries.
    "plot_gallery": "False" if os.environ.get("CI", "").lower() == "true" else "True",
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
    # Order the SECTIONS pedagogically (mirrors _GALLERY_SECTION_ORDER /
    # _fix_gallery_index_toctree below). Without this, sphinx-gallery sorts
    # sections alphabetically by directory name — putting "Advanced Topics"
    # first, which is the opposite of the newcomer-friendly flow we want.
    # Populated below ``_GALLERY_SECTION_ORDER`` is defined.
    "thumbnail_size": (320, 224),
    "default_thumb_file": None,
    "matplotlib_animations": False,
    "abort_on_example_error": False,
    "min_reported_time": 2,
}


# ── Gallery-index post-processor ────────────────────────────────────────────
# sphinx-gallery writes ``auto_examples/index.rst`` with ``:orphan:`` (so the
# page is unreachable from any sidebar) and dumps its toctree at the very
# bottom after the final category heading. This breaks the sidebar in two
# ways:
#
# 1. ``:orphan:`` keeps the gallery out of the main toctree → no sidebar
#    entry.
# 2. Content following a heading is a child of that heading, so every
#    gallery subsection ended up nested under "X-ray Emission".
#
# Fix: strip ``:orphan:`` so the gallery participates in the sidebar,
# move the toctree to the top of the file (right after the page title)
# so it's not a child of any heading, and reorder the entries
# pedagogically (onboarding → physics building blocks → observation
# layer → inference → applications) rather than alphabetical.
_GALLERY_SECTION_ORDER = (
    "quickstart",
    "workflows",
    "recipes",
    "sps",
    "sfh",
    "metallicity",
    "dust_attenuation",
    "dust_emission",
    "nebular",
    "igm",
    "agn",
    "radio",
    "xray",
    "photometry",
    "spectroscopy",
    "multiwavelength",
    "inference",
    "usecases",
    "advanced",
)

# Tell sphinx-gallery to emit gallery subsections in the same pedagogical
# order on the gallery landing page body (otherwise it falls back to
# alphabetical-by-directory and "Advanced Topics" lands first).
from sphinx_gallery.sorting import ExplicitOrder  # noqa: E402

sphinx_gallery_conf["subsection_order"] = ExplicitOrder(
    [f"../examples/{s}" for s in _GALLERY_SECTION_ORDER]
)


def _fix_gallery_index_toctree(app, *_args, **_kwargs):
    """Restructure ``auto_examples/index.rst`` for the main sidebar.

    Sphinx-gallery emits a flat file with ``:orphan:`` at the top, prose
    title further down, and one ``.. toctree::`` block at the very bottom
    (after a category heading). We need: (1) no ``:orphan:`` so the page
    appears in the sidebar; (2) a single toctree right after the page
    title, with entries reordered per ``_GALLERY_SECTION_ORDER``.

    Implemented line-by-line for robustness; regex was brittle against
    sphinx-gallery's quirky whitespace.
    """
    import re
    from pathlib import Path

    path = Path(app.srcdir) / "auto_examples" / "index.rst"
    if not path.exists():
        return

    lines = path.read_text().splitlines()

    # ── Step 1: find every "/auto_examples/<section>/index.rst" entry line
    # anywhere in the file, capture them, and remove them. Sphinx-gallery
    # writes these at the bottom; a previous broken post-processor run may
    # have moved them somewhere else. Either way: collect all, dedup,
    # remove from source.
    entry_re = re.compile(r"^(\s*)/auto_examples/([^/]+)/index\.rst\s*$")
    entries: dict[str, str] = {}  # section name → original line (indent preserved)
    out_lines = []
    for line in lines:
        m = entry_re.match(line)
        if m:
            entries.setdefault(m.group(2), line)
        else:
            out_lines.append(line)
    if not entries:
        # Nothing to fix yet; sphinx-gallery hasn't run.
        return

    # ── Step 2: drop any stray ``.. toctree::`` blocks (header + options).
    # A toctree block: line "``.. toctree::``" then 0+ indented option lines
    # (``:hidden:``, ``:includehidden:`` …) then a blank line. Without
    # entries (we already stripped those), the block is empty and should
    # not survive.
    cleaned = []
    i = 0
    while i < len(out_lines):
        line = out_lines[i]
        if line.strip() == ".. toctree::":
            # Skip the directive line and any directly-following option /
            # blank lines.
            i += 1
            while i < len(out_lines):
                nxt = out_lines[i]
                if not nxt.strip():
                    i += 1
                    continue
                # ``:option:`` lines start with whitespace + ":"
                if re.match(r"^\s+:[\w-]+:", nxt):
                    i += 1
                    continue
                break
            continue
        cleaned.append(line)
        i += 1

    # ── Step 3: drop the ``:orphan:`` directive so the page is in TOCs.
    cleaned = [ln for ln in cleaned if ln.strip() != ":orphan:"]
    # Trim leading blank lines we may have introduced.
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)

    # ── Step 4: build the new toctree with entries reordered.
    order = {name: i for i, name in enumerate(_GALLERY_SECTION_ORDER)}
    ordered_sections = sorted(
        entries.keys(),
        key=lambda s: (order.get(s, len(order)), s),
    )
    # ``:hidden:`` (no ``:includehidden:``) so the section list shows up
    # on the gallery page body but does NOT propagate into the parent
    # sidebar — the gallery is one dropdown, not 19 nested ones.
    toctree = [
        ".. toctree::",
        "   :hidden:",
        "",
    ] + [f"   /auto_examples/{s}/index.rst" for s in ordered_sections]

    # ── Step 5: re-emit. Splice the toctree in right after the page
    # title (the first "===" underline). The toctree must come after at
    # least one heading so sphinx parses it as page content, not
    # document-level metadata.
    title_under_idx = next(
        (
            j
            for j in range(1, len(cleaned))
            if cleaned[j].startswith("=") and cleaned[j].strip("=") == ""
        ),
        None,
    )
    if title_under_idx is None:
        # No title heading found — prepend toctree at top.
        new_lines = list(toctree) + [""] + cleaned
    else:
        new_lines = (
            cleaned[: title_under_idx + 1]
            + ["", *toctree, ""]
            + cleaned[title_under_idx + 1 :]
        )

    path.write_text("\n".join(new_lines) + "\n")


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
