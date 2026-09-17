#!/usr/bin/env python3
"""CI guard: version consistency between pyproject.toml and CITATION.cff.

The single source of truth for the project version is pyproject.toml. This guard
verifies that CITATION.cff's version field matches the version in pyproject.toml.

CITATION.cff is YAML consumed by GitHub and Zenodo and cannot derive the version
via Python, so it requires a manual copy. This guard turns that copy from a silent
failure mode into a red PR, ensuring the two versions stay in sync.

For each file mismatch found, the exit code is 1 and both versions are named.

Usage
-----
    python tools/check_version_single_source.py           # scan repo root
    python tools/check_version_single_source.py --root .  # same

Exit code 0 when versions match; 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: repo root).",
    )
    return parser.parse_args(argv)


def get_pyproject_version(root: Path) -> str | None:
    """Extract version from pyproject.toml."""
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data.get("project", {}).get("version")


def get_citation_version(root: Path) -> str | None:
    """Extract version from CITATION.cff."""
    citation_path = root / "CITATION.cff"
    if not citation_path.exists():
        return None

    content = citation_path.read_text(encoding="utf-8")
    # Match the version line in CITATION.cff
    match = re.search(r'^\s*version:\s*(\S+)', content, re.MULTILINE)
    return match.group(1) if match else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    pyproject_version = get_pyproject_version(args.root)
    citation_version = get_citation_version(args.root)

    if pyproject_version is None:
        print(f"ERROR: Could not find version in {args.root / 'pyproject.toml'}")
        return 1

    if citation_version is None:
        print(f"ERROR: Could not find version in {args.root / 'CITATION.cff'}")
        return 1

    if pyproject_version == citation_version:
        print(
            f"OK: pyproject.toml and CITATION.cff versions match ({pyproject_version})"
        )
        return 0

    print(
        f"FAIL: version mismatch\n"
        f"  pyproject.toml: {pyproject_version}\n"
        f"  CITATION.cff: {citation_version}\n"
        f"\nFix: Update CITATION.cff's version to match pyproject.toml."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
