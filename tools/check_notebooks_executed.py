#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fail when a published notebook was shipped having aborted part-way.

Why this exists
---------------

The reproduction notebooks stop on a missing input with ``raise SystemExit``,
which is the right call interactively -- it prints the one command that fixes it
instead of an eighty-line traceback. Under ``nbclient`` it is a trap:
**SystemExit is a clean stop, not an error.** Every cell after it is left
unexecuted, and the run reports success.

Measured on 01_cigale, whose 415 MB SSP grid is gitignored and so absent from a
fresh checkout: the notebook stopped at its second cell, the runner reported
*zero errors*, and it wrote a notebook carrying 5 of its 18 figures and none of
its 10 §-lines. Exit code 0.

Nothing else catches it. ``check_notebook_renders`` asks whether a render has
any figure -- 5 is not 0. ``check_figure_placement`` asks whether figures sit
beside the code that draws them -- with the later half of the notebook gone, the
surviving 5 are perfectly placed. Both pass. The published page is gutted.

The invariant here is the one that actually distinguishes the two cases: a
notebook is publishable only if every code cell in it ran. That is what
``execution_count`` records, and it cannot be faked by a partial run.

Usage
-----

::

    python tools/check_notebooks_executed.py
    python tools/check_notebooks_executed.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _published() -> list[Path]:
    """Reproduction notebooks only, and deliberately.

    These are written by ``nbclient`` in a single pass, so every code cell that
    ran carries an ``execution_count`` and the invariant is exact.

    The spine tutorials are **not** checked, because there the signal does not
    survive the pipeline: ``jupytext --sync`` regenerates the ``.ipynb`` from
    the newer ``.py`` and outputs are re-attached by matching cell source, which
    restores no count for a cell that produced no output. Measured on
    docs/spine/00_quickstart.ipynb, cells 1, 3 and 7 -- imports, an SSP load, a
    key split -- all show ``execution_count: None`` with zero outputs in a
    perfectly good render, and cell 3's ``load_ssp`` plainly ran because the
    cells after it print results that depend on it. Checking them would report
    six healthy tutorials as broken, and a guard that cries wolf gets muted.
    """
    return sorted((ROOT / "reproduction").glob("*/01_*.ipynb")) + sorted(
        (ROOT / "docs" / "reproduction").glob("*.ipynb")
    )


def unexecuted(nb: dict) -> list[int]:
    """Indices of non-empty code cells that carry no execution count."""
    out = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if not text.strip():
            continue
        if cell.get("execution_count") is None:
            out.append(i)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print every notebook checked")
    args = ap.parse_args(argv)

    problems: list[str] = []
    checked = 0
    for path in _published():
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path.relative_to(ROOT)}: unreadable ({exc})")
            continue
        checked += 1
        missing = unexecuted(nb)
        if args.list:
            print(f"  {str(path.relative_to(ROOT)):52} unexecuted: {len(missing)}")
            continue
        if missing:
            problems.append(
                f"{path.relative_to(ROOT)}: {len(missing)} code cell(s) never ran, "
                f"first at index {missing[0]} -- the run aborted (SystemExit reads as "
                f"success under nbclient). Generate the missing input named in that "
                f"cell and re-execute."
            )

    if args.list:
        return 0

    if problems:
        print(
            f"FAIL: {len(problems)} notebook(s) published from a partial run:\n", file=sys.stderr
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"OK: {checked} published notebooks ran every code cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
