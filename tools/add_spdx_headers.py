#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Ensure every Python source file under src/ and tests/ declares its license via SPDX.

Walks the configured roots and prepends ``# SPDX-License-Identifier: BSD-3-Clause``
to any ``.py`` file that does not already contain the SPDX marker. Empty
``__init__.py`` files (0 bytes) are skipped.

If a file starts with a ``#!`` shebang, the SPDX line is inserted after it.

Usage
-----
    python tools/add_spdx_headers.py            # apply edits in place
    python tools/add_spdx_headers.py --check    # CI mode: exit 1 if any file is missing the header

Modeled on tools/check_param_prefixes.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPDX_LINE = "# SPDX-License-Identifier: BSD-3-Clause\n"
SPDX_MARKER = "SPDX-License-Identifier"
DEFAULT_ROOTS = ("src/tengri", "tests")


def needs_header(path: Path) -> bool:
    """Return True if the file is a non-empty .py file lacking an SPDX marker."""
    if path.suffix != ".py":
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if not text.strip():
        return False
    return SPDX_MARKER not in text


def insert_header(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        new_text = lines[0] + SPDX_LINE + "".join(lines[1:])
    else:
        new_text = SPDX_LINE + text
    path.write_text(new_text, encoding="utf-8")


def iter_python_files(roots: tuple[str, ...]) -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    files: list[Path] = []
    for root in roots:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        files.extend(sorted(root_path.rglob("*.py")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not modify files; exit non-zero if any file lacks the SPDX header.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Repository-relative directory to scan (repeatable). Defaults: src/tengri, tests.",
    )
    args = parser.parse_args()

    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    missing = [p for p in iter_python_files(roots) if needs_header(p)]

    if args.check:
        if missing:
            print(f"{len(missing)} file(s) missing SPDX header:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 1
        print("All Python files declare SPDX-License-Identifier.")
        return 0

    for path in missing:
        insert_header(path)
    print(f"Added SPDX header to {len(missing)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
