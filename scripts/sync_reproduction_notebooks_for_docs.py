"""Refresh a reproduction notebook's source from its ``.py`` and publish it.

The reproduction comparisons keep three copies of every notebook:
``reproduction/<slug>/01_<slug>.py`` (jupytext percent, the source of truth),
``reproduction/<slug>/01_<slug>.ipynb`` (the render, with stored outputs), and
``docs/reproduction/<slug>.ipynb`` (what nbsphinx publishes). A prose-only edit
touches the first and must reach the other two without re-executing anything —
executing needs the external reference codes, their template grids, and in
CIGALE's case a 400 MB SSP repackaging, none of which CI has.

This script does that transplant: it copies each cell's *source* from the ``.py``
into the existing ``.ipynb``, leaving every stored output untouched, then copies
the result to ``docs/reproduction/``. It refuses to run when the cell layout
changed, because that means a code cell moved and the stored outputs no longer
belong to the cells holding them.

Usage::

    python scripts/sync_reproduction_notebooks_for_docs.py <slug> [<slug> ...]
    python scripts/sync_reproduction_notebooks_for_docs.py --all

See ``reproduction/CONTRACT.md`` §8 (publishing) and
``tests/contract/test_reproduction_docs_sync.py``, which pins the SHA-1 of every
embedded PNG across the source and docs copies.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRO_DIR = ROOT / "reproduction"
DOCS_DIR = ROOT / "docs" / "reproduction"


class LayoutChanged(RuntimeError):
    """Raised when the ``.py`` and ``.ipynb`` no longer share a cell layout."""


def _figure_hashes(nb) -> list[str]:
    """SHA-1 of every ``image/png`` output, in document order.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        A parsed notebook.

    Returns
    -------
    list of str
        One hex digest per embedded PNG.
    """
    out = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                out.append(hashlib.sha1(png.encode("utf-8")).hexdigest())
    return out


def transplant_source(py_path: Path, ipynb_path: Path) -> tuple[int, int]:
    """Copy cell source from ``py_path`` into ``ipynb_path``, keeping outputs.

    Parameters
    ----------
    py_path : Path
        The jupytext percent-format source of truth.
    ipynb_path : Path
        The rendered notebook whose stored outputs must survive.

    Returns
    -------
    tuple of (int, int)
        Number of cells updated, and number of embedded PNGs carried through.

    Raises
    ------
    LayoutChanged
        If cell count or cell types differ. Unlike the spine-notebook helper
        this is fatal rather than a silent fall back to an output-less
        notebook: dropping the figures here would sail past the docs-sync
        contract test only by breaking both copies at once.
    """
    import jupytext
    import nbformat

    fresh = jupytext.read(str(py_path))
    existing = nbformat.read(str(ipynb_path), as_version=4)

    before = _figure_hashes(existing)

    if len(existing.cells) != len(fresh.cells) or any(
        e.cell_type != n.cell_type for e, n in zip(existing.cells, fresh.cells)
    ):
        raise LayoutChanged(
            f"{py_path.name}: cell layout diverged from {ipynb_path.name} "
            f"({len(fresh.cells)} cells in .py vs {len(existing.cells)} in .ipynb). "
            "A code cell moved, so the stored outputs no longer match their "
            "cells. Re-render the notebook instead of transplanting."
        )

    for old_cell, new_cell in zip(existing.cells, fresh.cells):
        old_cell.source = new_cell.source

    nbformat.write(existing, str(ipynb_path))

    after = _figure_hashes(nbformat.read(str(ipynb_path), as_version=4))
    if before != after:
        raise LayoutChanged(
            f"{ipynb_path.name}: embedded figures changed during transplant "
            f"({len(before)} -> {len(after)}). Refusing to publish."
        )
    # A hash comparison alone passes trivially when both sides are empty, which
    # is exactly the state a half-finished render leaves behind: jupytext has
    # replaced the .ipynb but nbconvert has not written its outputs yet. Every
    # reproduction notebook has figures, so zero means "not rendered yet", and
    # publishing it would push an empty page to the docs site while the
    # docs-sync contract test stayed green -- it compares source against docs,
    # and two equally-empty notebooks agree.
    if not after:
        raise LayoutChanged(
            f"{ipynb_path.name} has no embedded figures. Every reproduction "
            "notebook has some, so this is an unrendered or half-rendered "
            "notebook. Refusing to publish; finish the render first."
        )
    return len(existing.cells), len(after)


def sync_slug(slug: str) -> int:
    """Transplant one comparison's source and publish it to ``docs/``.

    Parameters
    ----------
    slug : str
        Comparison folder name, e.g. ``"cigale"``.

    Returns
    -------
    int
        Process-style status: 0 on success, 1 on failure.
    """
    py_path = REPRO_DIR / slug / f"01_{slug}.py"
    ipynb_path = REPRO_DIR / slug / f"01_{slug}.ipynb"
    docs_path = DOCS_DIR / f"{slug}.ipynb"

    for path in (py_path, ipynb_path):
        if not path.is_file():
            print(f"  {slug}: missing {path.relative_to(ROOT)}", file=sys.stderr)
            return 1

    try:
        n_cells, n_figs = transplant_source(py_path, ipynb_path)
    except LayoutChanged as exc:
        print(f"  {slug}: {exc}", file=sys.stderr)
        return 1

    shutil.copyfile(ipynb_path, docs_path)
    print(
        f"  {slug}: {n_cells} cells refreshed, {n_figs} figures preserved "
        f"-> {docs_path.relative_to(ROOT)}"
    )
    return 0


def discover_slugs() -> list[str]:
    """Return every comparison slug that has both a source and a docs copy."""
    return sorted(
        p.name
        for p in REPRO_DIR.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and (p / f"01_{p.name}.py").is_file()
        and (DOCS_DIR / f"{p.name}.ipynb").is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slugs", nargs="*", help="comparison slugs to sync")
    parser.add_argument("--all", action="store_true", help="sync every comparison")
    args = parser.parse_args()

    slugs = discover_slugs() if args.all else args.slugs
    if not slugs:
        parser.error("give at least one slug, or --all")

    print(f"syncing {len(slugs)} comparison(s):")
    return max(sync_slug(s) for s in slugs)


if __name__ == "__main__":
    raise SystemExit(main())
