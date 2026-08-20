#!/usr/bin/env python3
"""Sync spine Jupytext sources into docs/spine/*.ipynb for Sphinx + nbsphinx.

Steps (per notebook):
1. ``jupytext --sync`` on ``notebooks/<slug>.py`` (creates/updates the paired
   ``notebooks/<slug>.ipynb``; that path is gitignored at repo root).
2. Copy the synced ``.ipynb`` into ``docs/spine/<slug>.ipynb``.

If the paired file is missing after sync, falls back to:

   jupytext notebooks/<slug>.py --to ipynb -o docs/spine/<slug>.ipynb

3. Run ``jupyter nbconvert --to notebook --inplace`` on each ``docs/spine/*.ipynb``
   (format check; nbsphinx uses nbconvert when building HTML).

4. Normalize markdown headings: exactly one H1 ``# <slug>`` (file stem); any other
   top-level ``#`` lines become ``##`` so Sphinx does not list them in the sidebar.
   ``EXPERIMENTAL_SLUGS`` are exempt from the retitle and keep their own H1.

Covers both the numbered spine (``SPINE_SLUGS`` -> ``docs/spine/``) and the
standalone demonstrations (``EXPERIMENTAL_SLUGS`` -> ``docs/spine/experimental/``).
The latter were published via the ``docs/index.md`` toctree while absent from this
script, so both were hand-copied and both drifted from their sources; see #1506
and the ``tools/check_notebook_renders.py`` guard that now pins the invariant.

Run from repository root::

    python scripts/sync_spine_notebooks_for_docs.py

Or via ``make -C docs spine-ipynb``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SPINE_SLUGS = [
    "00_quickstart",
    "01_why_jax",
    "02_sed_anatomy",
    "03_discovering_the_menu",
    "04_building_models",
    "05_fitting_photometry",
    "06_fitting_spectroscopy",
    "07_joint_photo_spec",
    "08_emission_lines",
    "10_fastspecfit_joint_fit",
    "11_catalog_fits",
    "12_simulation_populations",
    "09_parameter_sweeps",
]

# Human-readable titles for Sphinx sidebar (keyed by slug).
SPINE_TITLES: dict[str, str] = {
    "00_quickstart": "Quickstart",
    "01_why_jax": "Why JAX",
    "02_sed_anatomy": "SED Anatomy",
    "03_discovering_the_menu": "Discovering the Menu",
    "04_building_models": "Building Models",
    "05_fitting_photometry": "Fitting Photometry",
    "06_fitting_spectroscopy": "Fitting Spectroscopy",
    "07_joint_photo_spec": "Joint Photometry + Spectroscopy",
    "08_emission_lines": "Emission Lines",
    "10_fastspecfit_joint_fit": "Joint Fit: Photometry + Lines",
    "11_catalog_fits": "Catalog Fits in Parallel",
    "12_simulation_populations": "Forward-Modeling Simulation Populations",
    "09_parameter_sweeps": "Parameter Sweeps",
}

# Published under docs/spine/experimental/ -- the "Experimental" toctree at the end
# of docs/index.md. Their SOURCES live flat in notebooks/, so the output path and
# the slug differ, which is exactly why a flat SPINE_SLUGS list missed them: both
# renders were hand-copied and both drifted from their sources (#1506).
#
# Unlike the numbered spine, these keep their OWN H1. They are standalone research
# demonstrations rather than a sequence, so the notebook's title is the useful
# sidebar entry, and duplicating it in SPINE_TITLES would just be a second copy to
# drift.
EXPERIMENTAL_SLUGS = [
    "stochastic_sfh_recovery",
    "multimodel_bma_candels",
    "jwst_nonparametric_fits",
    # Apple-GPU guide. Its render is produced on Apple Silicon in a separate
    # environment (jax-mps + jax 0.10), because the notebook selects
    # JAX_PLATFORMS=mps and cannot execute anywhere else. check_notebook_renders
    # compares the render against the source and requires a figure; it does not
    # re-execute, so a hand-run render is the intended path here rather than a
    # gap in automation.
    "apple_mps",
    # NVIDIA/CUDA guide. Unlike apple_mps this one executes in the ordinary
    # environment (jax[cuda12] on the pinned JAX), so its render is produced by
    # running the notebook on a CUDA box rather than by hand.
    "nvidia_cuda",
]
EXPERIMENTAL_SUBDIR = "experimental"


def _demote_extra_h1_lines(text: str) -> str:
    """Turn ``# foo`` into ``## foo`` except the first line (already the doc title)."""
    lines = text.split("\n")
    fixed: list[str] = []
    for i, line in enumerate(lines):
        if i == 0:
            fixed.append(line)
            continue
        stripped = line.lstrip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            indent = line[: len(line) - len(stripped)]
            fixed.append(indent + "#" + stripped)
        else:
            fixed.append(line)
    return "\n".join(fixed)


def excluded_spine_slugs(conf_path: Path) -> set[str]:
    """Spine slugs that ``docs/conf.py`` keeps out of the published build.

    A notebook listed in ``exclude_patterns`` still exists on disk (and on
    GitHub) but Sphinx emits no HTML page for it, so any link pointing at it
    from a *published* notebook is dead. Read the list rather than duplicating
    it here: a hardcoded copy silently rots the next time someone publishes or
    hides a notebook, and the resulting dead link is invisible (see
    :func:`check_published_links`).
    """
    text = conf_path.read_text(encoding="utf-8")
    if "exclude_patterns" not in text:
        # Fail loudly. Returning an empty set here would mean "nothing is
        # excluded", and every dead link would sail through the check below.
        raise RuntimeError(
            f"{conf_path}: no exclude_patterns -- cannot tell published from hidden"
        )
    # Match nested paths too ("spine/experimental/foo.ipynb"), then key on the
    # stem: callers compare against notebook slugs, which are flat. A pattern of
    # [^"/]+ silently skipped every nested entry, so an excluded experimental
    # notebook would have read as published and its inbound links as live (#1506).
    return {m.rsplit("/", 1)[-1] for m in re.findall(r'"spine/([^"]+)\.ipynb"', text)}


def normalize_spine_links(ipynb_path: Path, excluded: set[str]) -> tuple[int, int]:
    """Point sibling-notebook links at something the docs build actually emits.

    In ``notebooks/`` a link like ``[quickstart](00_quickstart.py)`` is correct --
    the sibling really is a ``.py`` file, and it resolves on GitHub. In ``docs/``
    only the rendered ``.ipynb`` exists, and nbsphinx resolves link targets as
    local FILES (it does not accept the extensionless MyST doc form inside a
    notebook), so the ``.py`` target 404s and nbsphinx warns ``localfile``.
    Rewriting at sync time keeps each context correct instead of forcing one to
    carry the other's spelling.

    Two cases, because retargeting alone is not enough:

    * target is **published** -- rewrite ``.py`` to ``.ipynb``.
    * target is **excluded** by ``conf.py`` -- drop the link, keep the text.
      Retargeting these would be worse than leaving them: the ``.py`` form at
      least makes nbsphinx warn ``localfile``, whereas ``.ipynb`` names a file
      that *is* on disk, so the warning stops while the href stays dead. Loud
      broken beats silent broken.

    Only ``SPINE_SLUGS`` targets are touched, so links to genuine ``.py`` files
    (scripts, benchmarks) are left alone.

    Returns
    -------
    tuple of (int, int)
        Counts of ``(retargeted, de-linked)`` links.
    """
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    n_retarget = n_delink = 0

    # Rewrite per markdown cell, never over the raw file. The de-link pattern
    # spans characters ("[...](...)"), and a regex like that turned loose on the
    # JSON text will happily match across string and object boundaries and
    # corrupt the notebook.
    delink_res = {
        slug: re.compile(r"\[([^\]\n]*)\]\(" + re.escape(slug) + r"\.(?:py|ipynb)\)")
        for slug in SPINE_SLUGS
        if slug in excluded
    }

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        body = "".join(src) if isinstance(src, list) else str(src)
        original = body

        for slug in SPINE_SLUGS:
            if slug in delink_res:
                body, n = delink_res[slug].subn(r"\1", body)
                n_delink += n
            else:
                target = f"]({slug}.py)"
                n_retarget += body.count(target)
                body = body.replace(target, f"]({slug}.ipynb)")

        if body != original:
            cell["source"] = body

    if n_retarget or n_delink:
        ipynb_path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return n_retarget, n_delink


def check_published_links(spine_out: Path, excluded: set[str]) -> list[str]:
    """Report links in published notebooks that resolve to no built page.

    Sphinx cannot catch this. nbsphinx's ``localfile`` warning fires when the
    target FILE is missing from disk; a link to an excluded notebook names a
    file that is committed and present, so the build stays green while the
    rendered href 404s. This is the only check that sees it.
    """
    problems: list[str] = []
    link_re = re.compile(r"\]\(([^)]+)\)")

    published = [(slug, spine_out / f"{slug}.ipynb") for slug in SPINE_SLUGS]
    published += [
        (slug, spine_out / EXPERIMENTAL_SUBDIR / f"{slug}.ipynb") for slug in EXPERIMENTAL_SLUGS
    ]

    for slug, nb_path in published:
        if slug in excluded:
            continue  # not published; its own links never render
        if not nb_path.is_file():
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source", "")
            body = "".join(src) if isinstance(src, list) else str(src)
            for raw in link_re.findall(body):
                target = raw.split("#")[0].split("?")[0].strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                stem = Path(target).stem
                if stem in excluded:
                    why = "excluded from build; href is dead"
                elif target.endswith(".py") and stem in SPINE_SLUGS:
                    why = "docs ship .ipynb; missed retarget"
                else:
                    continue
                problems.append(f"{slug}.ipynb -> {target} ({why})")
    return problems


def normalize_markdown_headings(ipynb_path: Path, slug: str) -> None:
    """One H1 title in the first cell; demote other top-level ``#`` lines to ``##``."""
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    title = SPINE_TITLES.get(slug, slug)
    title_heading = f"# {title}"
    first_md = True
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if first_md:
            m = re.search(r"^#", text, flags=re.MULTILINE)
            if m:
                text = text[m.start() :]
            new_text, n_sub = re.subn(
                r"^#[^\n]*\n+",
                title_heading + "\n\n",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n_sub == 0:
                new_text = title_heading + "\n\n" + text.lstrip("\n")
            cell["source"] = _demote_extra_h1_lines(new_text)
            first_md = False
        else:
            lines = text.split("\n")
            fixed: list[str] = []
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    indent = line[: len(line) - len(stripped)]
                    fixed.append(indent + "#" + stripped)
                else:
                    fixed.append(line)
            cell["source"] = "\n".join(fixed)
    ipynb_path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _merge_source_preserve_outputs(py_path: Path, out_ipynb: Path) -> None:
    """Refresh ``out_ipynb`` source from ``py_path`` while keeping outputs.

    The committed ``docs/spine/*.ipynb`` is the source of execution outputs
    (figures, prints, HTML reprs). The repo's ``notebooks/*.py`` is the
    source of code text. On CI ``notebooks/*.ipynb`` is absent (gitignored),
    so a naive ``jupytext --sync`` followed by a copy would overwrite the
    committed outputs with an empty notebook. Instead:

    1. Read the existing ``out_ipynb`` via ``nbformat`` (keeps outputs).
    2. Parse a fresh notebook from ``py_path`` via ``jupytext`` (just source).
    3. If the cell layout matches (count + types), copy each new cell's
       ``source`` into the existing cell, preserving outputs.
    4. If the layout diverged (developer added/removed cells), fall back to
       the fresh notebook with empty outputs — the dev must re-run locally
       and recommit.
    """
    import jupytext as _jp
    import nbformat as _nbf

    new = _jp.read(str(py_path))

    if not out_ipynb.is_file():
        _nbf.write(new, str(out_ipynb))
        return

    existing = _nbf.read(str(out_ipynb), as_version=4)

    same_layout = len(existing.cells) == len(new.cells) and all(
        e.cell_type == n.cell_type for e, n in zip(existing.cells, new.cells)
    )

    if not same_layout:
        # Cell structure changed — keep new source, lose outputs.
        _nbf.write(new, str(out_ipynb))
        return

    for old_cell, new_cell in zip(existing.cells, new.cells):
        old_cell.source = new_cell.source

    _nbf.write(existing, str(out_ipynb))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nb_root = root / "notebooks"
    spine_out = root / "docs" / "spine"
    spine_out.mkdir(parents=True, exist_ok=True)

    excluded = excluded_spine_slugs(root / "docs" / "conf.py")

    # (slug, output path, whether to force the H1 to SPINE_TITLES). The slug stays
    # the flat notebook stem in both cases because link rewriting matches on the
    # source filename, which is flat even for the nested renders.
    targets = [(slug, spine_out / f"{slug}.ipynb", True) for slug in SPINE_SLUGS]
    targets += [
        (slug, spine_out / EXPERIMENTAL_SUBDIR / f"{slug}.ipynb", False)
        for slug in EXPERIMENTAL_SLUGS
    ]

    for slug, out_ipynb, retitle in targets:
        py_path = nb_root / f"{slug}.py"

        if not py_path.is_file():
            print(f"error: missing {py_path}", file=sys.stderr)
            return 1

        out_ipynb.parent.mkdir(parents=True, exist_ok=True)

        # Preserve committed outputs (they're how nbsphinx renders figures
        # on CI, where a freshly-generated ipynb would have none).
        _merge_source_preserve_outputs(py_path, out_ipynb)

        if retitle:
            normalize_markdown_headings(out_ipynb, slug)
        n_retarget, n_delink = normalize_spine_links(out_ipynb, excluded)

        notes = []
        if n_retarget:
            notes.append(f"{n_retarget} link(s) retargeted to .ipynb")
        if n_delink:
            notes.append(f"{n_delink} link(s) de-linked (target not published)")
        suffix = f" ({'; '.join(notes)})" if notes else ""
        print(f"synced {slug} -> {out_ipynb.relative_to(root)}{suffix}")

    problems = check_published_links(spine_out, excluded)
    if problems:
        print(
            f"\nerror: {len(problems)} link(s) in published notebooks resolve to no built page:",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
