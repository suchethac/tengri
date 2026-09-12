#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fail when a committed reproduction render no longer matches its source.

Why this exists
----------------

``scripts/render_reproduction_notebook.py`` stamps every render it produces
with ``metadata["tengri_render"] = {"source_sha256", "executed_at",
"tengri_version"}`` -- the SHA-256 of the source ``.py``'s code cells at the
moment it was executed. Nothing else pins that a committed
``reproduction/<slug>/01_<slug>.ipynb`` was actually produced from the
``.py`` sitting beside it on disk *today*: a source edit lands, the render is
not re-run, and the stale ``.ipynb`` (and its stale ``docs/reproduction/``
copy) keeps shipping -- exactly the failure mode
``tests/contract/test_reproduction_docs_sync.py`` was written for, one layer
upstream of where that test can see it (it only compares the two committed
copies to *each other*, never to the source that supposedly produced them).

This guard closes that gap: it recomputes the source hash from the
checked-out ``.py`` (via the same stdlib-only
``tools._repro_render_shared.code_cell_source_sha256`` the render script
stamps with) and compares it to the stamp.

The pattern mirrors ``tools/check_repro_status.py``: most reproduction
notebooks predate this stamp, so an **unstamped** render only warns (exit 0)
-- this guard must not block before every notebook has been re-rendered
once. A **stamped-but-stale** render always fails, in both modes: a stamp
that disagrees with its own source is worse than no stamp, since it looks
current. ``--strict`` additionally requires every notebook to carry a stamp
(for a future state where every render has been through the render script at
least once).

Usage::

    python tools/check_repro_render_fresh.py            # warn-only on unstamped (exit 0)
    python tools/check_repro_render_fresh.py --strict    # also require a stamp on every notebook
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._repro_render_shared import code_cell_source_sha256


def _repro_slugs(root: Path) -> list[str]:
    """Comparison slugs under ``<root>/reproduction/`` with both a ``.py`` and ``.ipynb``."""
    repro_dir = root / "reproduction"
    if not repro_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in repro_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and (p / f"01_{p.name}.py").is_file()
        and (p / f"01_{p.name}.ipynb").is_file()
    )


def check_slug(slug: str, *, root: Path = ROOT) -> str | None:
    """Freshness verdict for one comparison.

    Parameters
    ----------
    slug : str
        Comparison folder name under ``<root>/reproduction/``.
    root : Path, optional
        Repository root, overridable for testing against a scratch tree.

    Returns
    -------
    {"fresh", "unstamped", "stale"}
        ``"stale"`` means the stamp exists and disagrees with the source's
        current hash; ``"unstamped"`` means no ``tengri_render`` stamp is
        present; ``"fresh"`` means the stamp matches.
    """
    slug_dir = root / "reproduction" / slug
    py_path = slug_dir / f"01_{slug}.py"
    ipynb_path = slug_dir / f"01_{slug}.ipynb"

    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    stamp = nb.get("metadata", {}).get("tengri_render")
    if not stamp:
        return "unstamped"

    current_hash = code_cell_source_sha256(py_path)
    if stamp.get("source_sha256") != current_hash:
        return "stale"
    return "fresh"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict", action="store_true", help="require a stamp on every reproduction notebook"
    )
    args = parser.parse_args(argv)

    slugs = _repro_slugs(ROOT)
    if not slugs:
        print(f"FAIL: no reproduction comparisons found under {ROOT / 'reproduction'}")
        return 1

    stale: list[str] = []
    unstamped: list[str] = []
    fresh = 0

    for slug in slugs:
        verdict = check_slug(slug, root=ROOT)
        if verdict == "stale":
            stale.append(slug)
        elif verdict == "unstamped":
            unstamped.append(slug)
        else:
            fresh += 1

    if stale:
        print(f"FAIL: {len(stale)} reproduction render(s) are stale:\n")
        for slug in stale:
            print(
                f"  {slug}: reproduction/{slug}/01_{slug}.ipynb's tengri_render stamp no "
                f"longer matches its source. Re-render with "
                f"'python scripts/render_reproduction_notebook.py {slug}'."
            )
        print()

    if args.strict and unstamped:
        print(f"FAIL (strict): {len(unstamped)} reproduction notebook(s) have no render stamp:")
        for slug in unstamped:
            print(f"  {slug}")
        return 1

    if stale:
        return 1

    if unstamped:
        print(
            f"reproduction render freshness: {len(unstamped)} notebook(s) have no "
            "tengri_render stamp yet (pass --strict to require one):"
        )
        for slug in unstamped:
            print(f"  {slug}")

    print(f"OK: {fresh}/{len(slugs)} reproduction render(s) are stamped and fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
