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
    "10_fastspecfit_joint_fit": "FastSpecFit Joint Fit (Photometry + Lines)",
    "09_parameter_sweeps": "Parameter Sweeps",
}


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

    for slug in SPINE_SLUGS:
        py_path = nb_root / f"{slug}.py"
        out_ipynb = spine_out / f"{slug}.ipynb"

        if not py_path.is_file():
            print(f"error: missing {py_path}", file=sys.stderr)
            return 1

        # Preserve committed outputs (they're how nbsphinx renders figures
        # on CI, where a freshly-generated ipynb would have none).
        _merge_source_preserve_outputs(py_path, out_ipynb)

        normalize_markdown_headings(out_ipynb, slug)

        print(f"synced {slug} -> {out_ipynb.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
