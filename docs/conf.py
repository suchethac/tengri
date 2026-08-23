"""Sphinx configuration for tengri documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


def _ensure_nbconvert_templates_visible() -> None:
    """Put the installed nbconvert's data directory on ``JUPYTER_PATH``.

    nbsphinx exports notebooks through nbconvert, which finds its templates by
    scanning ``JUPYTER_PATH`` -- not by looking beside the module it imported.
    When nbconvert resolves outside the environment running sphinx (a venv
    without its own copy falls through to a system or Anaconda install), that
    scan misses and the build dies with::

        ValueError: No template sub-directory with name 'rst' found

    which names templates and says nothing about the actual cause. It kills the
    whole build, after the gallery has already been walked.

    Deriving it here rather than in each caller is deliberate: ``make html``,
    ``sphinx-autobuild`` and ``tools/regen_gallery.py`` are three entry points
    and only one of them can be told to remember an environment variable.

    A no-op when nbconvert is absent, when its data directory is not laid out as
    ``<prefix>/share/jupyter``, or when ``JUPYTER_PATH`` is already set.
    """
    if os.environ.get("JUPYTER_PATH"):
        return
    try:
        import nbconvert
    except ImportError:
        return
    import pathlib

    for parent in pathlib.Path(nbconvert.__file__).resolve().parents:
        candidate = parent / "share" / "jupyter"
        if (candidate / "nbconvert" / "templates").is_dir():
            os.environ["JUPYTER_PATH"] = str(candidate)
            return


_ensure_nbconvert_templates_visible()

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

# Scoped regen escape hatch: set ``TENGRI_GALLERY_ONLY`` to a comma-separated
# list of example basenames (e.g. ``plot_qpah_sweep,plot_tdust_vs_lir``) to
# execute *only* those scripts on the next ``make html``. Used after a library
# change that alters a handful of figures, so the heavy NUTS/VI scripts (and
# everything else) are left as-is instead of re-running. Empty/unset keeps the
# default disk-driven behavior.
import re as _re

_only = os.environ.get("TENGRI_GALLERY_ONLY", "").strip()
_only_alt = "|".join(_re.escape(b) for b in _only.split(",") if b.strip())

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    # Run any plot_*.py whose basename is NOT in _DO_NOT_EXECUTE. Sphinx-
    # gallery applies this with ``re.search`` against the *full source path*
    # (e.g. ``/Users/.../examples/<group>/plot_foo.py``), so ``^plot_`` would
    # never match. Anchor at start of string with ``^``, run the negative-
    # lookahead against the whole path to exclude any basename already in
    # ``_DO_NOT_EXECUTE``, then ``.*plot_<...>.py$`` to pin the filename.
    "filename_pattern": (
        rf"(?:{_only_alt})\.py$" if _only_alt else rf"^(?!.*(?:{_skip_alt})).*plot_[^/]+\.py$"
    ),
    # No ignore_pattern: the gallery is forward-model only (2026-07 overhaul
    # removed inference/fit scripts). All examples render quickly. If a future
    # example is too heavy to build, delete or lighten it rather than
    # hiding it here: hidden scripts bit-rot invisibly because CI never
    # executes the gallery.
    "download_all_examples": False,
    # Locally we execute (default). On CI (e.g. GitHub Actions sets CI=true) we
    # use the pre-rendered docs/auto_examples/ that the developer committed so
    # the cloud build doesn't need SSP grids, RELAGN, or ~20 GB of optional
    # data.
    #
    # To refresh a render, use `python tools/regen_gallery.py <basename>` and
    # commit what it writes. This comment used to say "run make html and commit
    # the regenerated auto_examples/", which destroys stale pages (#1236):
    # sphinx-gallery rewrites a page whose source md5 no longer matches its
    # stamp, but `filename_pattern` above stops it being executed, so it comes
    # back without the output execution produced. Measured on a single-example
    # build while 60 examples were stale: 195 files, 45,204 deletions, exit 0.
    #
    # A page that is already fresh is not rewritten at all — a full build over
    # a fresh gallery was measured to change nothing. So the damage is
    # proportional to the drift, and the freshness gate that keeps drift at
    # zero is also what keeps a plain build safe. regen_gallery.py is still the
    # right tool while anything is stale: it restores every non-target page.
    "plot_gallery": "False" if os.environ.get("CI", "").lower() == "true" else "True",
    # Skip re-execution of examples whose source + md5 haven't changed since the
    # last build. Cuts incremental regen from ~25 min to seconds when only one
    # example changed.
    "run_stale_examples": True,
    # Parallel execution is opt-in via env var (defaults to 1 / serial) because
    # sphinx-gallery's parallel mode requires `joblib` which isn't a docs-build
    # dependency in CI. The ignore_pattern above already hides the OOM-prone
    # NUTS/VI scripts so 2-4 is safe locally if you have joblib installed.
    "parallel": int(os.environ.get("TENGRI_GALLERY_PARALLEL", "1")),
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
# layer → applications) rather than alphabetical.
_GALLERY_SECTION_ORDER = (
    # Onboarding: where every astronomer should start.
    "quickstart",
    "recipes",
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
    # Science applications.
    "usecases",
    "advanced",
    # Headline end-to-end demonstrations.
    "showcase",
)

# Tell sphinx-gallery to emit gallery subsections in the same pedagogical
# order on the gallery landing page body (otherwise it falls back to
# alphabetical-by-directory and "Advanced Topics" lands first).
from sphinx_gallery.sorting import ExplicitOrder

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

    Executed pages carry ``.. image-sg::`` (not ``.. image::``). Two
    hazards, both handled here:

    * A plain ``.. image::`` must never be injected into a page that
      already has ``.. image-sg::`` — matching only the plain form
      double-injected a figure into every executed page (the 2026-07
      double-figure bug).
    * On a full-execution build this one-shot hook can fire while
      sphinx-gallery is still writing per-script ``.. image-sg::``
      directives, so a page injected here may *later* also get an
      ``image-sg``. So the pass is idempotent and self-healing: if a
      page ends up with BOTH, the injected plain block is stripped.
    """
    from pathlib import Path

    auto = Path(app.srcdir) / "auto_examples"
    if not auto.exists():
        return
    # Dedup pass: strip an injected plain ``.. image::`` block from any
    # page that also carries the canonical ``.. image-sg::`` (race repair).
    deduped = 0
    for rst in auto.glob("*/plot_*.rst"):
        text = rst.read_text()
        if ".. image-sg::" not in text or "\n.. image:: images/sphx_glr_" not in text:
            continue
        lines = text.splitlines()
        out, i, changed = [], 0, False
        while i < len(lines):
            if lines[i].startswith(".. image:: images/sphx_glr_"):
                j = i + 1
                while j < len(lines) and (lines[j].startswith("   ") or not lines[j].strip()):
                    j += 1
                i = j
                changed = True
                continue
            out.append(lines[i])
            i += 1
        if changed:
            rst.write_text("\n".join(out) + "\n")
            deduped += 1
    if deduped:
        print(f"[conf.py] stripped double-injected .. image:: from {deduped} RSTs")

    fixed = 0
    for rst in auto.glob("*/plot_*.rst"):
        text = rst.read_text()
        if ".. image::" in text or ".. image-sg::" in text:
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
            if (
                not inserted
                and i + 1 < len(lines)
                and lines[i + 1].strip()
                and set(lines[i + 1].strip()) <= {"="}
            ):
                out.append(lines[i + 1])
                out += [
                    "",
                    f".. image:: {img}",
                    f"   :alt: {stem.replace('_', ' ')}",
                    "   :class: sphx-glr-single-img",
                    "",
                ]
                for j in range(i + 2, len(lines)):
                    out.append(lines[j])
                inserted = True
                break
        if inserted:
            rst.write_text("\n".join(out) + "\n")
            fixed += 1
    if fixed:
        print(f"[conf.py] injected .. image:: directives into {fixed} skipped RSTs")


# ── Per-object citation labels ──────────────────────────────────────────────
# NAMING_CONTRACT / docstring-standard mandate numpydoc ``References`` sections
# using ``.. [1]``, ``.. [2]``. docutils reads those as *footnotes*, whose
# labels must be unique per document — but autodoc renders dozens of objects
# onto one page (api/core, api/models, ...), so every docstring redefines
# ``[1]``. That produced 88 "Duplicate explicit target name" warnings.
#
# src/ has 582 such definitions across 143 files plus 114 inline ``[N]_``
# references. Renaming them by hand is infeasible and would contradict the
# documented citation standard. Switching napoleon out for the numpydoc
# extension does fix the collisions (it mangles labels the same way), but with
# ``numpydoc_show_class_members = False`` it drops the class-docstring
# ``Attributes`` descriptions from the rendered page entirely — measured: the
# text survives only in the viewcode source listing. A fix that deletes
# documentation is not a fix.
#
# So do the label mangling here and leave the docstrings alone: rewrite
# ``.. [N]`` and ``[N]_`` to carry the fully-qualified object name, which is
# unique per object by construction. Rendering is unchanged — readers still
# see "[1]" — only the internal target id differs.
_CITATION_DEF = _re.compile(r"^(\s*)\.\.\s+\[(\d+)\]", _re.MULTILINE)
_CITATION_REF = _re.compile(r"\[(\d+)\]_")


def _uniquify_citation_labels(app, what, name, obj, options, lines):
    """Namespace numeric citation labels by owning object.

    Turns ``.. [1]`` into ``.. [tengri.Fitter-1]`` and ``[1]_`` into
    ``[tengri.Fitter-1]_`` for one docstring, so two objects rendered onto the
    same page cannot collide. No-op for docstrings without numeric citations.
    """
    if not lines:
        return
    text = "\n".join(lines)
    if "[1]" not in text and "[2]" not in text:
        return  # cheap bail-out: the overwhelming majority of docstrings
    slug = name.replace("%", "-")
    new = _CITATION_DEF.sub(rf"\1.. [{slug}-\2]", text)
    new = _CITATION_REF.sub(rf"[{slug}-\1]_", new)
    if new != text:
        lines[:] = new.split("\n")


# ── Component reference, generated from the live registries ─────────────────
# `docs/api/discovery.rst` autodocs the ``list_*`` *functions*, so the menu they
# return is reachable only by running Python. This writes their *contents* into
# the build, which is the difference between documenting that a catalog exists
# and publishing the catalog.
#
# Generated per build and gitignored, never committed. The worst docs failure
# this repo has had (#1236, see the filename_pattern note above) is a committed
# artifact silently rewritten into a hollow one; a page rebuilt from the live
# registry on every build has no stale state to rot into.
#
# The census is *derived* — every ``tengri.list_*`` that exists — rather than a
# hand-kept list, because a hand-kept list is exactly how a newly registered
# family goes unpublished without anyone noticing. `_REGISTRY_ORDER` only sets
# the reading order; anything absent from it is still emitted, at the end.
_GENERATED_DIR = _pl.Path(__file__).parent / "_generated"

_REGISTRY_ORDER = (
    "list_components",
    "list_recipes",
    "list_sfh_models",
    "list_metallicity_modes",
    "list_known_ssps",
    "list_age_kernels",
    "list_dust_models",
    "list_dust_laws",
    "list_dust_emission_models",
    "list_nebular_backends",
    "list_shock_models",
    "list_agn_models",
    "list_agn_blocks",
    "list_xray_models",
    "list_radio_models",
    "list_radio_blocks",
    "list_igm_models",
    "list_inference_methods",
    "list_instruments",
)

# Menus too long to print in full. Named on the page with their size and the
# call that returns them, rather than dropped: a family missing with no trace is
# indistinguishable from one that was never registered.
_TOO_LONG_TO_PRINT = ("list_filters",)


def _write_parameter_tables(app=None):
    """Render per-domain parameter tables from _GROUP_STRUCTURAL_KEYS."""
    import pathlib
    import subprocess
    import sys

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "generate_parameter_tables.py"
    result = subprocess.run([sys.executable, str(script)], cwd=str(repo_root))
    if result.returncode != 0:
        raise RuntimeError(f"Parameter table generator failed with code {result.returncode}")


def _write_component_reference(app=None):
    """Render every live registry to ``_generated/component_tables.rst``."""
    import tengri

    listers = [n for n in dir(tengri) if n.startswith("list_") and n != "list_all"]
    ordered = [n for n in _REGISTRY_ORDER if n in listers]
    ordered += sorted(n for n in listers if n not in _REGISTRY_ORDER)

    chunks = []
    for name in ordered:
        try:
            table = getattr(tengri, name)()
        except Exception as exc:  # a registry that cannot list is a real defect
            raise RuntimeError(f"tengri.{name}() failed while building the docs: {exc}") from exc
        if not len(table):
            continue
        heading = name.removeprefix("list_").replace("_", " ")
        chunks.append(f"{heading}\n{'-' * len(heading)}\n\n``tengri.{name}()``\n")
        if name in _TOO_LONG_TO_PRINT:
            chunks.append(
                f"{len(table)} entries — too many to print here. "
                f"Call ``tengri.{name}()`` for the full table, or narrow it, "
                f"e.g. ``tengri.{name}().filter(survey='SDSS')``.\n"
            )
        else:
            chunks.append(table.to_rst())
        chunks.append("")

    _GENERATED_DIR.mkdir(exist_ok=True)
    (_GENERATED_DIR / "component_tables.rst").write_text("\n".join(chunks), encoding="utf-8")


def setup(app):
    # Before anything is read, so components.md can include the result.
    app.connect("builder-inited", _write_component_reference)
    app.connect("builder-inited", _write_parameter_tables)
    # Priority 1000 runs *after* sphinx-gallery's own builder-inited handler.
    app.connect("builder-inited", _fix_gallery_index_toctree, priority=1000)
    # env-before-read-docs fires AFTER sphinx-gallery has generated the
    # per-script RSTs but BEFORE sphinx parses any source, which is exactly
    # the window we need to mutate the RSTs on disk.
    app.connect("env-before-read-docs", _inject_missing_image_directives)
    # Runs after napoleon has converted the numpydoc References section to RST.
    app.connect("autodoc-process-docstring", _uniquify_citation_labels)


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

# Render numpydoc ``Attributes`` sections as ``:ivar:`` info-fields rather than
# standalone ``.. attribute::`` directives.
#
# Many tengri classes list an attribute in the class docstring's ``Attributes``
# section *and* implement it as a real property with its own docstring. With
# the default (False), napoleon emits a ``py:attribute`` object for the former
# while ``:members:`` emits one for the latter, so the same name is registered
# twice on the same page — "duplicate object description of
# tengri.Parameters.all_params, other instance in api/core". That accounted for
# 115 of the 416 warnings; ``Parameters`` alone lists 7 such attributes and
# produced exactly 7 warnings. ``:ivar:`` fields are not separate objects, so
# nothing collides and both descriptions still render.
napoleon_use_ivar = True

# -- Warning suppression -----------------------------------------------------
#
# Exactly two categories, both meaning the same benign thing: "a References
# section lists a source that no inline ``[N]_`` marker cites".
#
# The docstring standard requires a numpydoc ``References`` section whenever a
# formula or algorithm comes from a paper. It does not require citing each
# entry inline in the prose, and for most entries that would read badly — the
# section *is* the citation. docutils disagrees and warns once per uncited
# entry: 74 as citations (autodoc docstrings, after the label-uniquifying hook
# below) and 50 as footnotes (sphinx-gallery example headers).
#
# Deliberately NOT suppressed: ``docutils``. That category carries genuinely
# malformed markup — block quotes without a blank line, short title
# underlines, unterminated inline markup — which are real defects and are
# fixed rather than hidden.
suppress_warnings = [
    "ref.citation",
    "ref.footnote",
    # The landing page supplies its <h1> inside the styled hero div
    # (``<h1 class="tg-hero__title">``) rather than as a markdown heading, so
    # MyST reports "Document headings start at H2". The heading exists; it is
    # just HTML so it can carry the hero styling.
    "myst.header",
    # Ambiguity only. ``nitpicky`` is off, so Sphinx does not report unresolved
    # Python references; the sole ``ref.python`` warning left is "more than one
    # target found for cross-reference 'n_grid'". That comes from the shape
    # annotations the docstring standard mandates —
    # ``ndarray, shape (n_grid,)`` — where the shape variable happens to share
    # a name with a property documented on ForwardModel, SEDModel and
    # Parameters (the same value, delegated down the chain). Sphinx picks one
    # and links it. The alternatives are dropping the mandated shape
    # annotation or hiding two of the three properties; neither is an
    # improvement.
    "ref.python",
]

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
    # Registry tables written by _write_component_reference each build. They
    # are ``.. include::``-ed into components.md, so as a *source document*
    # the fragment is an orphan — a warning, which -W makes a build failure.
    "_generated",
    # Everything contributor-only: agent-authored plans and specs (was
    # `superpowers/{plans,specs}`, needing two entries of its own) plus the six
    # narrative sections listed further down. One rule now covers them and
    # anything else contributor-facing that lands here later.
    #
    # Both forms on purpose. The bare name prunes the directory during
    # traversal; the `**` form is what `dev/**` and `adr/**` already use to keep
    # their contents from building as orphans. The bare entry predates the
    # directory existing, so it has never actually been exercised.
    "internal",
    "internal/**",
    "**.ipynb_checkpoints",
    # Synthesizer reproduction notebook kept out of the published docs by
    # request (removed from the reproduction toctree). Excluded so Sphinx
    # does not warn about an orphaned document. Restore alongside the
    # toctree entry.
    "reproduction/synthesizer.ipynb",
    # Sphinx-gallery internal outputs that shouldn't be picked up as source
    "auto_examples/index.rst.new",
    "sg_execution_times.rst",
    # sphinx-gallery also writes a per-section timing page. Only the top-level
    # one was excluded, so the per-section copies were still parsed — and each
    # links to every example in its section via a ``sphx_glr_...`` label. Those
    # labels only exist for scripts the build actually executed, and the build
    # deliberately reuses committed renders (``plot_gallery`` is False on CI,
    # and locally ``_DO_NOT_EXECUTE`` skips anything already rendered). The
    # result was 18 "undefined label" warnings for timing pages nothing links
    # to. Exclude them for the same reason the top-level one is excluded.
    "auto_examples/*/sg_execution_times.rst",
    "auto_examples/**/*.py",
    "auto_examples/**/*.md5",
    "auto_examples/**/*.codeobj.json",
    "auto_examples/**/*.zip",
    "auto_examples/**/*.ipynb",
    # ``getting_started/``, ``inference/``, ``advanced/``, ``developer/``,
    # ``user/`` and ``recipes/`` each had a line here. They were narrative
    # sections superseded by the repo-root notebooks/ spine, hidden one at a
    # time over 2026-06/07 — every call sound on its own, but the result was
    # six rules and 19 maintained pages sitting in ``docs/`` that no reader
    # could reach. They now live under ``internal/``, which the single
    # ``internal`` entry near the top of this list already covers.
    # (``forward_model/`` and ``fitting/`` are gone entirely: the former's one
    # page is published as ``forward_model.md``, the latter held only a toctree.)
    "dev/**",
    # Architecture Decision Records are contributor-facing design history, not
    # part of the published user docs. They live on disk / GitHub but are
    # excluded here so they don't build as orphans (each emitted a
    # "not in any toctree" warning).
    "adr/**",
    "NEBULAR_REFACTOR.md",
    "_notebooks/**",
    # Not part of the published sidebar (content folded into index.md or omitted)
    "changelog.md",
    # Spine notebooks not currently in the published sidebar (08 emission
    # lines + 09 parameter sweeps were the "physics deep dives" section,
    # dropped from the index in the 2026-05 polish pass).
    "spine/08_emission_lines.ipynb",
    "spine/09_parameter_sweeps.ipynb",
    # ``spine/05_adding_a_model.ipynb`` was listed here. Unlike the two above it
    # was also absent from ``SPINE_SLUGS`` in
    # ``scripts/sync_spine_notebooks_for_docs.py``, so nothing synced it and
    # ``tools/check_notebook_renders.py`` did not guard it -- a hand-copy of
    # ``notebooks/05_adding_a_model.py`` that no longer built, with nothing
    # keeping it in step with its source. That is the drift #1506 added the
    # guard to stop, and it had already started: the copy needed editing by hand
    # alongside its source. The copy is deleted; the source is untouched. To
    # publish this page again, add its slug to ``SPINE_SLUGS`` and re-run the
    # sync -- which is how every other spine notebook gets here.
    # Fitting-photometry spine notebook hidden from the published sidebar
    # (2026-06): the quickstart already covers a full photometry fit +
    # posterior, so this longer treatment is redundant. Inbound prose links
    # in 04/06/07 now point at the quickstart instead.
    "spine/05_fitting_photometry.ipynb",
    # The Guides section was hidden from the published sidebar in 2026-06. Two
    # of the original four are published again: ``method_selection.md`` (README
    # points at it) and ``known_limitations.md`` (a runtime warning in
    # ``sed_model.py`` tells users to read it). The other two -- ``recipes.md``
    # and ``joint_fitting.md`` -- had entries here; they now live under
    # ``internal/``, covered by the rule at the top of this list, since
    # ``spine/07_joint_photo_spec`` and ``tengri.list_recipes()`` already serve
    # readers and nothing outside ``docs/`` linked either page.
    #
    # ``known_bugs.md`` moved to ``docs/dev/`` (excluded via the ``dev/**``
    # rule above). ``changelog.md`` stays at ``docs/`` root; both are named in
    # check_british_spelling and check_doc_examples exemption lists.
]
