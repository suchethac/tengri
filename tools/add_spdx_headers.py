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
import re
import sys
from pathlib import Path

SPDX_MARKER = "SPDX-License-Identifier"
DEFAULT_ROOTS = ("src/tengri", "tests")

#: Pattern for a declared identifier, e.g. ``# SPDX-License-Identifier: BSD-3-Clause``.
SPDX_RE = re.compile(rf"{SPDX_MARKER}:\s*(\S+)")


def project_license() -> str:
    """The licence ``pyproject.toml`` declares, so the two cannot drift.

    Hardcoding the expected identifier here would mean a relicence has to be
    made in two places and silently passes if it is made in one.
    """
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^license\s*=\s*\{\s*text\s*=\s*\"([^\"]+)\"", text, re.M)
    if match is None:
        raise SystemExit("pyproject.toml declares no license = {text = ...}; cannot check")
    return match.group(1)


EXPECTED_LICENSE = project_license()
SPDX_LINE = f"# {SPDX_MARKER}: {EXPECTED_LICENSE}\n"


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


def declared_license(path: Path) -> str | None:
    """The identifier a file declares, or None when it declares none."""
    if path.suffix != ".py":
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = SPDX_RE.search(text)
    return match.group(1) if match else None


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
    files = iter_python_files(roots)
    missing = [p for p in files if needs_header(p)]
    # Presence was the only thing checked until now, so four files declaring a
    # licence this project does not use passed for as long as they existed:
    # three MIT and one GPL-3.0-or-later, all of them tengri's own tests. A
    # guard that asks only whether a line exists cannot see what it says.
    mismatched = [
        (p, lic)
        for p in files
        if (lic := declared_license(p)) is not None and lic != EXPECTED_LICENSE
    ]

    if args.check:
        if missing:
            print(f"{len(missing)} file(s) missing SPDX header:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
        if mismatched:
            print(
                f"{len(mismatched)} file(s) declare a licence other than "
                f"{EXPECTED_LICENSE} (from pyproject.toml):",
                file=sys.stderr,
            )
            for path, lic in mismatched:
                print(f"  {path}  declares {lic}", file=sys.stderr)
            print(
                "\nIf the file is this project's own work, correct the header. If it "
                "is genuinely third-party under that licence, it does not belong in "
                "a BSD-3-Clause package without a deliberate decision -- which is "
                "why --fix will not rewrite these.",
                file=sys.stderr,
            )
        if missing or mismatched:
            return 1
        print(
            f"All {len(files)} Python files declare "
            f"{SPDX_MARKER}: {EXPECTED_LICENSE}."
        )
        return 0

    # Only insert where there is nothing. A wrong identifier is a statement
    # about provenance and rewriting it automatically could erase a real
    # third-party attribution, so those are reported and left alone.
    for path in missing:
        insert_header(path)
    print(f"Added SPDX header to {len(missing)} file(s).")
    if mismatched:
        print(
            f"Left {len(mismatched)} file(s) with a non-{EXPECTED_LICENSE} "
            "identifier untouched; run --check to list them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
