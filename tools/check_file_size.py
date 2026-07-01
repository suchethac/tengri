#!/usr/bin/env python3
"""CI guard for file size compliance (max 800 lines per src/**/*.py file).

Files exceeding 800 lines must be explicitly allowlisted in
file_size_allowlist.json at their baseline line count. This guard enforces:

1. No new files exceed 800 lines (add to allowlist if refactoring is needed).
2. Allowlisted files must not grow beyond their recorded baseline.
3. Allowlisted files that shrink to <= 800 lines must be removed from the
   allowlist (ratchet must tighten).

Usage
-----
    python tools/check_file_size.py

Exit code 0 if all checks pass; non-zero with violations listed otherwise.

The allowlist is managed by humans; this script is a compliance gate only.
Violations can be fixed by:
- Refactoring large files into smaller modules.
- Adding new files to the allowlist (temporarily) with a tracking note.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = REPO_ROOT / "tools" / "file_size_allowlist.json"
MAX_LINES = 800


def count_lines(path: Path) -> int:
    """Count non-empty lines in a Python file."""
    try:
        content = path.read_text(encoding="utf-8")
        return len(content.splitlines())
    except Exception:
        return 0


def main() -> int:
    """Run the file size guard.

    Returns
    -------
    int
        0 if all checks pass; 1 if violations found.
    """
    # Load allowlist
    try:
        with open(ALLOWLIST_PATH) as f:
            allowlist = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load allowlist at {ALLOWLIST_PATH}: {e}", file=sys.stderr)
        return 1

    violations = []

    # Scan all .py files in src/
    src_dir = REPO_ROOT / "src"
    if not src_dir.exists():
        print(f"ERROR: src/ directory not found at {src_dir}", file=sys.stderr)
        return 1

    for py_file in sorted(src_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(REPO_ROOT))
        current_lines = count_lines(py_file)

        if rel_path in allowlist:
            baseline_lines = allowlist[rel_path]
            # Check if allowlisted file grew beyond baseline
            if current_lines > baseline_lines:
                violations.append(
                    (
                        "grew",
                        rel_path,
                        f"grew from {baseline_lines} to {current_lines} lines",
                    )
                )
            # Check if allowlisted file can now be removed (shrunk to <= 800)
            elif current_lines <= MAX_LINES:
                violations.append(
                    (
                        "shrunk",
                        rel_path,
                        f"shrunk to {current_lines} lines; remove from allowlist",
                    )
                )
        else:
            # File not in allowlist
            if current_lines > MAX_LINES:
                violations.append(
                    (
                        "new_violation",
                        rel_path,
                        f"{current_lines} lines (max {MAX_LINES}, add to allowlist if needed)",
                    )
                )

    # Check for orphaned allowlist entries (files deleted or moved)
    for allowlisted_path in allowlist:
        full_path = REPO_ROOT / allowlisted_path
        if not full_path.exists():
            violations.append(
                (
                    "orphaned",
                    allowlisted_path,
                    "file no longer exists; remove from allowlist",
                )
            )

    # Report findings
    if violations:
        print(f"FAIL: {len(violations)} file size violation(s)\n")
        by_type = {}
        for vtype, path, msg in violations:
            if vtype not in by_type:
                by_type[vtype] = []
            by_type[vtype].append((path, msg))

        for vtype in sorted(by_type.keys()):
            type_label = {
                "new_violation": "New file exceeds 800 lines",
                "grew": "Allowlisted file grew",
                "shrunk": "Allowlisted file shrunk (should remove)",
                "orphaned": "Orphaned allowlist entry",
            }.get(vtype, vtype)
            print(f"  [{type_label}]")
            for path, msg in sorted(by_type[vtype]):
                print(f"    {path}: {msg}")
        print()
        return 1

    print(
        f"OK: all {len(list(src_dir.rglob('*.py')))} Python files comply "
        f"with {MAX_LINES}-line limit (or allowlisted at baseline)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
