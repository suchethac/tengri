# SPDX-License-Identifier: BSD-3-Clause
"""Partition a sorted list of file paths into N contiguous slices.

This script enables parallel CI legs by splitting test files into disjoint,
contiguous portions of a sorted path list. Each part is a contiguous block,
ensuring no gaps or overlaps, and enabling deterministic file distribution
across parallel job legs.

Output format: one path per line by default, or space-joined on one line
with --join. Exits with code 2 if the requested part is empty (no files
would be assigned to that leg).
"""

from __future__ import annotations

import argparse
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
    return parser.parse_args(argv)


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

    # Output the files
    if args.join:
        print(" ".join(str(f) for f in part_files))
    else:
        for f in part_files:
            print(str(f))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
