# SPDX-License-Identifier: BSD-3-Clause
"""Partition a sorted list of file paths into N contiguous slices.

This script enables parallel CI legs by splitting test files into disjoint,
contiguous portions of a sorted path list. Each part is a contiguous block,
ensuring no gaps or overlaps, and enabling deterministic file distribution
across parallel job legs.

With --durations <json> the split is instead balanced by measured duration:
files are assigned by greedy longest-processing-time (duration descending,
ties by path; each goes to the currently lightest part, ties by lowest part
index). A file missing from the ledger gets the ledger's median duration.
Within a part the paths are sorted. The ledger for tests/regression/bug is
tools/ci_bug_durations.json ({path: seconds}); regenerate it by running, from
the repo root, with PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu TENGRI_DATA_DIR=<data>
TENGRI_DISABLE_JAX_CACHE=1:

    python -m pytest tests/regression/bug -q -p no:cacheprovider -n 4 \\
        --dist=loadfile -o addopts="--tb=short --strict-markers \\
        -m 'not crossval and not slow and not benchmark and not population_fit'" \\
        --durations=0 --durations-min=0 --ignore=tests/crossval \\
        --ignore=tests/regression/paper/test_draine2021_pah_loader.py \\
        --ignore=tests/regression/synthesizer_parity/test_nebular_continuum.py \\
        --ignore=tests/regression/synthesizer_parity/test_nebular_fesc.py > durations.txt

then sum the setup+call+teardown lines per file and write them rounded to 0.1 s
with sorted keys (a per-file sum of the three phases).

Output format: one path per line by default, or space-joined on one line
with --join. Exits with code 2 if the requested part is empty (no files
would be assigned to that leg).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Partition sorted file paths into contiguous slices for parallel CI legs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=False,
        help="Root directory to search for files (required unless --paths is used).",
    )
    parser.add_argument(
        "--part",
        type=int,
        required=False,
        help="Which part to return (1-indexed).",
    )
    parser.add_argument(
        "--of",
        type=int,
        required=False,
        help="Total number of parts.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="test_*.py",
        help="Glob pattern for files to match (default: test_*.py).",
    )
    parser.add_argument(
        "--join",
        action="store_true",
        help="Output files on a single space-joined line instead of one per line.",
    )
    parser.add_argument(
        "--durations",
        type=Path,
        default=None,
        help="JSON ledger {path: seconds}; balance parts by duration (greedy LPT) "
        "instead of contiguous equal-count slices.",
    )
    return parser.parse_args(argv)


def balance_by_duration(
    files: list[Path], ledger: dict[str, float], parts: int
) -> list[list[Path]]:
    """Greedy longest-processing-time assignment; deterministic.

    Files absent from the ledger take the ledger's median duration.
    """
    median = statistics.median(ledger.values()) if ledger else 1.0
    weight = {f: float(ledger.get(str(f), median)) for f in files}
    ordered = sorted(files, key=lambda f: (-weight[f], str(f)))
    loads = [0.0] * parts
    buckets: list[list[Path]] = [[] for _ in range(parts)]
    for f in ordered:
        idx = min(range(parts), key=lambda i: (loads[i], i))
        loads[idx] += weight[f]
        buckets[idx].append(f)
    return [sorted(b) for b in buckets]


def _emit(part_files: list[Path], join: bool) -> None:
    """Print the part, space-joined or one path per line."""
    if join:
        print(" ".join(str(f) for f in part_files))
    else:
        for f in part_files:
            print(str(f))


def main(argv: list[str] | None = None) -> int:
    """Partition and output files for a given part."""
    args = parse_args(argv)

    # Find all matching files and sort them
    root = args.root
    if not root.is_dir():
        print(f"::error::root directory not found: {root}", file=sys.stderr)
        return 1

    # Use rglob with maxdepth simulation via glob
    all_files = sorted(root.glob(args.pattern))

    if not all_files:
        print(f"::error::no files matching {args.pattern} in {root}", file=sys.stderr)
        return 1

    if args.durations is not None:
        try:
            ledger = json.loads(args.durations.read_text())
        except (OSError, ValueError) as exc:
            print(
                f"::error::cannot read durations ledger {args.durations}: {exc}", file=sys.stderr
            )
            return 1
        part_files = balance_by_duration(all_files, ledger, args.of)[args.part - 1]
        if not part_files:
            print(f"::error::empty part: part {args.part} of {args.of} is empty", file=sys.stderr)
            return 2
        _emit(part_files, args.join)
        return 0

    # Calculate the size of each part
    total_files = len(all_files)
    part_size = (total_files + args.of - 1) // args.of  # Ceiling division

    # Calculate start and end indices for this part (1-indexed)
    part_idx = args.part - 1  # Convert to 0-indexed
    start_idx = part_idx * part_size
    end_idx = min(start_idx + part_size, total_files)

    # Check if this part would be empty
    if start_idx >= total_files:
        print(f"::error::empty part: part {args.part} of {args.of} is empty", file=sys.stderr)
        return 2

    # Get the files for this part (contiguous slice)
    part_files = all_files[start_idx:end_idx]

    _emit(part_files, args.join)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
