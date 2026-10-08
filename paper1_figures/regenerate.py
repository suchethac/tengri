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

Exit status: 0 when every family produced its figures, ``3`` when some family
could not run for want of inputs (it names them), and 1 when one failed. Only
0 means the whole set was rebuilt -- a skip is not a success.

Each family prints one ``FIGURE <name> ok|missing: <why>`` line per figure it
owns, and the count below is taken from those lines, not from the family's
exit code alone: a figure that was refused is counted as not produced.

Figures land in ``paper1_figures/figures/`` and are NOT committed: ``fig*.pdf``
is excluded by .gitignore and ``tools/check_tracked_not_ignored.py`` enforces
it. The committed artifact is the notebook plus the data it reads.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import FIGURE_LINE, SKIPPED

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"
FIGURES = HERE / "figures"


def notebooks() -> list[Path]:
    return sorted(p for p in NOTEBOOKS.glob("*.py") if not p.name.startswith("_"))


def figure_lines(stdout: str) -> list[tuple[str, str | None]]:
    """Parse a family's ``FIGURE`` lines into ``(name, reason)`` pairs.

    ``reason`` is ``None`` for a figure the family produced.
    """
    found: list[tuple[str, str | None]] = []
    for line in stdout.splitlines():
        if not line.startswith(f"{FIGURE_LINE} "):
            continue
        _, name, rest = [*line.split(" ", 2), ""][:3]
        if rest == "ok":
            found.append((name, None))
        else:
            found.append((name, rest.removeprefix("missing: ") or "not produced"))
    return found


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
    skipped: list[str] = []
    produced: list[str] = []
    not_produced: list[tuple[str, str, str]] = []
    for path in chosen:
        print(f"--- {path.name}")
        done = subprocess.run(
            [sys.executable, str(path)], cwd=HERE.parent, capture_output=True, text=True
        )
        print(done.stdout.rstrip())
        lines = figure_lines(done.stdout)
        produced += [name for name, reason in lines if reason is None]
        not_produced += [(path.name, name, reason) for name, reason in lines if reason]
        if done.returncode == SKIPPED:
            # Not a success. A family that could not run for want of inputs
            # produced no figure, and reporting that as zero is how this
            # command came to promise more than it delivered.
            skipped.append(path.name)
        elif done.returncode != 0:
            failures.append(
                (path.name, done.stderr.strip().splitlines()[-1] if done.stderr else "?")
            )
            print(done.stderr, file=sys.stderr)

    after = {p.name for p in FIGURES.glob("*.pdf")}
    print(f"\nfigures produced: {len(produced)}, not produced: {len(not_produced)}")
    print(f"figures written: {len(after)} ({len(after - before)} new this run)")
    for name in sorted(after):
        print(f"  {name}")
    if not_produced:
        print(f"\n{len(not_produced)} figure(s) refused or missing:")
        for family, name, reason in not_produced:
            print(f"  {family}: {name}: {reason}")
    if skipped:
        print(f"\n{len(skipped)} family(ies) skipped for want of inputs:")
        for name in skipped:
            print(f"  {name}")
    if failures:
        print(f"\n{len(failures)} notebook(s) failed:", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1
    return SKIPPED if (skipped or not_produced) else 0


if __name__ == "__main__":
    raise SystemExit(main())
