#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Regenerate every figure in Paper I.

One command, so that "can a reader reproduce the figures" has a yes-or-no
answer rather than a per-figure investigation:

    python paper1_figures/regenerate.py

Each notebook under ``notebooks/`` is a jupytext percent file, which is also an
ordinary Python script, so this runs them directly and needs neither jupyter
nor nbconvert. A notebook that cannot be run headless is not reproducible, so
running them is the test.

Figures land in ``paper1_figures/figures/`` and are NOT committed: ``fig*.pdf``
is excluded by .gitignore and ``tools/check_tracked_not_ignored.py`` enforces
it. The committed artifact is the notebook plus the data it reads.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"
FIGURES = HERE / "figures"


def notebooks() -> list[Path]:
    return sorted(p for p in NOTEBOOKS.glob("*.py") if not p.name.startswith("_"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="substring of a notebook name to run alone")
    args = parser.parse_args(argv)

    chosen = [p for p in notebooks() if not args.only or args.only in p.name]
    if not chosen:
        print(f"no notebooks matched {args.only!r} in {NOTEBOOKS}", file=sys.stderr)
        return 2

    FIGURES.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in FIGURES.glob("*.pdf")}
    failures: list[tuple[str, str]] = []
    for path in chosen:
        print(f"--- {path.name}")
        done = subprocess.run(
            [sys.executable, str(path)], cwd=HERE.parent, capture_output=True, text=True
        )
        if done.returncode != 0:
            failures.append(
                (path.name, done.stderr.strip().splitlines()[-1] if done.stderr else "?")
            )
            print(done.stdout)
            print(done.stderr, file=sys.stderr)
        else:
            print(done.stdout.rstrip())

    after = {p.name for p in FIGURES.glob("*.pdf")}
    print(f"\nfigures written: {len(after)} ({len(after - before)} new this run)")
    for name in sorted(after):
        print(f"  {name}")
    if failures:
        print(f"\n{len(failures)} notebook(s) failed:", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
