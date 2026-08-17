#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fail when a published notebook piles its figures into one cell.

Why this exists
---------------

``check_notebook_renders.py`` asks whether a render carries *any* figure, which
catches the ``MPLBACKEND=Agg`` failure of #1506. It does not ask *where* the
figures landed, and that is a separate, silent failure.

matplotlib-inline 0.1.6 against IPython 8.27 breaks the per-cell flush. Every
reproduction notebook opens with

    get_ipython().run_line_magic("matplotlib", "inline")

precisely to guarantee inline embedding; under the broken pairing that line
*suppresses* it instead, and figures accumulate until the next explicit
``plt.show()``. Measured on 01_agnfitter: 20 figures beside the 20 cells that
draw them became 17 piled into 3. The notebook still exited 0 with every
assertion passing and a plausible total figure count, so nothing in the build
noticed. On the published page the reader scrolls eleven code cells with no
output and then hits a wall of images, which is the whole value of a worked
comparison destroyed by a dependency version.

``docs/requirements.txt`` floors matplotlib-inline at 0.1.7 to stop it
recurring. This check is the assertion that the floor is doing its job, because
the failure is invisible to every other guard.

The rule
--------

A notebook that draws in three or more cells must show its figures in at least
half that many cells. Deliberately loose: a cell may legitimately emit several
figures (a loop over models), and a helper may draw without the ``plt.subplots``
marker. It is tuned to catch total collapse -- 18 figures in one cell -- not to
police layout. Measured against the tree at the time of writing, every healthy
notebook shows at least as many cells as it draws in, so the margin is wide.

Usage
-----

::

    python tools/check_figure_placement.py           # fail on collapse
    python tools/check_figure_placement.py --list    # print the census
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Cells that construct a figure. A helper that wraps plt.subplots is missed,
# which is why the threshold below is a ratio and not equality.
_MAKES_FIGURE = re.compile(r"plt\.subplots|plt\.figure|pyplot\.subplots")

MIN_DRAWING_CELLS = 3
MIN_SHOW_RATIO = 0.5


def _published() -> list[Path]:
    """Every notebook that ships to the docs site."""
    paths = sorted((ROOT / "reproduction").glob("*/01_*.ipynb"))
    paths += sorted((ROOT / "docs" / "spine").glob("*.ipynb"))
    paths += sorted((ROOT / "docs" / "spine").glob("*/*.ipynb"))
    paths += sorted((ROOT / "docs" / "reproduction").glob("*.ipynb"))
    return paths


def census(nb: dict) -> tuple[int, int, int]:
    """``(cells that draw, cells that show, total figures)``."""
    draws = shows = figures = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if _MAKES_FIGURE.search(text):
            draws += 1
        n = sum(
            1 for out in (cell.get("outputs") or []) if "image/png" in (out.get("data") or {})
        )
        figures += n
        if n:
            shows += 1
    return draws, shows, figures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print the census for every notebook")
    args = ap.parse_args()

    problems: list[str] = []
    checked = 0
    for path in _published():
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path.relative_to(ROOT)}: unreadable ({exc})")
            continue
        draws, shows, figures = census(nb)
        checked += 1
        if args.list:
            print(f"  {str(path.relative_to(ROOT)):52} draws {draws:3d}  shows {shows:3d}  figs {figures:3d}")
            continue
        if draws >= MIN_DRAWING_CELLS and shows < draws * MIN_SHOW_RATIO:
            problems.append(
                f"{path.relative_to(ROOT)}: draws in {draws} cells but shows figures in only "
                f"{shows} ({figures} total) -- the per-cell flush is off. Check that "
                f"matplotlib-inline is >= 0.1.7 (docs/requirements.txt) and re-execute."
            )

    if args.list:
        return 0

    if problems:
        print(f"FAIL: {len(problems)} notebook(s) collapsed their figures:\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"OK: {checked} published notebooks place their figures beside the code that draws them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
