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
        # plot_hierarchical also hits an upstream stochastic-SFH JAX-tracing
        # issue in model.mock() / predict_observables under field SFH — needs
        # a library fix before re-enabling.
        r"plot_(population_scaling|hierarchical|hierarchical_convergence|prior_posterior_compare)\.py$"
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
    # Onboarding: where every astronomer should start.
    "quickstart",
    "recipes",
    "workflows",
    # The stellar engine: what produces the continuum.
    "sps",
    "sfh",
    "metallicity",
    # ISM processing: what happens between the stars and us.
    # Order mirrors the radiative-transfer pipeline:
    #   stellar continuum --> nebular reprocessing at the source
    #   --> dust attenuation along the line of sight
    #   --> dust thermal re-emission set by energy balance.
    "nebular",
    "dust_attenuation",
    "dust_emission",
    # AGN: the alternative engine.
    "agn",
    # Wavelength extensions: long- and short-λ companions of the optical
    # SED, plus the cosmological line-of-sight absorption.
    "radio",
    "xray",
    "igm",
    # Observation layer: how the SED couples to instruments.
    "photometry",
    "spectroscopy",
    "multiwavelength",
    # Inference and science applications.
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
    """Restructure ``auto_examples/index.rst`` and orphan its subsections.

    Goals:
      1. The gallery landing page (``auto_examples/index``) participates
         in the global toctree, so it appears as a single sidebar entry.
      2. The per-section index pages (``auto_examples/quickstart/index``
         and friends) are marked ``:orphan:`` so Furo's global toctree
         walker doesn't pull them into the sidebar as children of the
         gallery entry. Users still reach them via the thumbnail grid
         on the gallery landing page and via sphinx-gallery's per-script
         prev/next nav.
      3. No ``.. toctree::`` block in the landing page — that block was
         the channel through which subsections leaked into the sidebar.

    Side-effect: each subsection ``index.rst`` gets ``:orphan:`` injected
    at the top.
    """
    import re
    from pathlib import Path

    auto = Path(app.srcdir) / "auto_examples"
    path = auto / "index.rst"
    if not path.exists():
        return

    lines = path.read_text().splitlines()

    # Strip every "/auto_examples/<section>/index.rst" entry line and
    # remember which sections are present (so we can orphan them below).
    entry_re = re.compile(r"^(\s*)/auto_examples/([^/]+)/index\.rst\s*$")
    sections: set[str] = set()
    out_lines = []
    for line in lines:
        m = entry_re.match(line)
        if m:
            sections.add(m.group(2))
        else:
            out_lines.append(line)
    if not sections:
        return  # sphinx-gallery hasn't run yet

    # Drop any ``.. toctree::`` blocks (header + indented options + blanks).
    cleaned = []
    i = 0
    while i < len(out_lines):
        line = out_lines[i]
        if line.strip() == ".. toctree::":
            i += 1
            while i < len(out_lines):
                nxt = out_lines[i]
                if not nxt.strip():
                    i += 1
                    continue
                if re.match(r"^\s+:[\w-]+:", nxt):
                    i += 1
                    continue
                break
            continue
        cleaned.append(line)
        i += 1

    # Drop ``:orphan:`` on the LANDING page so it stays in the sidebar.
    cleaned = [ln for ln in cleaned if ln.strip() != ":orphan:"]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)

    path.write_text("\n".join(cleaned) + "\n")

    # Inject ``:orphan:`` into each per-section index so sphinx doesn't
    # walk them into the global sidebar and doesn't warn about pages
    # missing from any toctree.
    for section in sections:
        sub = auto / section / "index.rst"
        if not sub.exists():
            continue
        text = sub.read_text()
        if text.lstrip().startswith(":orphan:"):
            continue
        sub.write_text(":orphan:\n\n" + text.lstrip())


def _inject_missing_image_directives(app, *_args, **_kwargs):
    """Inject ``.. image::`` blocks into RSTs that lack one.

    Sphinx-gallery only writes the image directive into a script's RST
    when it *executes* the script. ``filename_pattern`` here skips any
    script whose ``images/sphx_glr_<stem>_001.png`` is already on disk
    (the speed-up that keeps docs build under 5 min). So skipped
    scripts produce RSTs without the directive, and the deployed HTML
    for those pages silently loses its figure.

    For every ``plot_*.rst`` that lacks ``.. image::`` but DOES have a
    matching image on disk, inject a standard sphinx-gallery image
    block right after the section title underline.
    """
    from pathlib import Path

    auto = Path(app.srcdir) / "auto_examples"
    if not auto.exists():
        return
    fixed = 0
    for rst in auto.glob("*/plot_*.rst"):
        text = rst.read_text()
        if ".. image::" in text:
            continue
        stem = rst.stem
        img = f"images/sphx_glr_{stem}_001.png"
        if not (rst.parent / img).exists():
            continue
        lines = text.splitlines()
        out = []
        inserted = False
        for i, line in enumerate(lines):
            out.append(line)
            if (not inserted and i + 1 < len(lines)
                    and lines[i + 1].strip()
                    and set(lines[i + 1].strip()) <= {"="}):
                out.append(lines[i + 1])
                out += ["",
                        f".. image:: {img}",
                        f"   :alt: {stem.replace('_', ' ')}",
                        "   :class: sphx-glr-single-img",
                        ""]
                for j in range(i + 2, len(lines)):
                    out.append(lines[j])
                inserted = True
                break
        if inserted:
            rst.write_text("\n".join(out) + "\n")
            fixed += 1
    if fixed:
        print(f"[conf.py] injected .. image:: directives into {fixed} skipped RSTs")


def setup(app):
    # Priority 1000 runs *after* sphinx-gallery's own builder-inited handler.
    app.connect("builder-inited", _fix_gallery_index_toctree, priority=1000)
    # env-before-read-docs fires AFTER sphinx-gallery has generated the
    # per-script RSTs but BEFORE sphinx parses any source, which is exactly
    # the window we need to mutate the RSTs on disk.
    app.connect("env-before-read-docs", _inject_missing_image_directives)

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
html_js_files = ["two-mode-theme.js"]

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
    # Spine notebooks not currently in the published sidebar (08 emission
    # lines + 09 parameter sweeps were the "physics deep dives" section,
    # dropped from the index in the 2026-05 polish pass).
    "spine/08_emission_lines.ipynb",
    "spine/09_parameter_sweeps.ipynb",
]
