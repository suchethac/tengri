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
    python tools/check_file_size.py                    # gate (CI mode)
    python tools/check_file_size.py --fix              # update pins in place
    python tools/check_file_size.py --fix --allow-new  # also add NEW >800 files

Exit code 0 if all checks pass; non-zero with violations listed otherwise.

``--fix`` mechanically applies the bookkeeping fixes (like ``ruff --fix``):
grown pins update to the measured size, shrunk/orphaned entries are removed.
It deliberately does NOT add new >800-line files unless ``--allow-new`` is
passed — a brand-new oversized module is a design decision, not bookkeeping.
Run it before committing any change that grows an allowlisted file; three
merges shipped with red pins in one week because this was manual (#1167,
#1204, #1188).

Violations can also be fixed by refactoring large files into smaller modules
— the allowlist is a burn-down list, not a quota.
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


def main(argv=None) -> int:
    """Run the file size guard.

    Parameters
    ----------
    argv : list of str, optional
        CLI args (default ``sys.argv[1:]``): ``--fix`` and ``--allow-new``.

    Returns
    -------
    int
        0 if all checks pass (or ``--fix`` repaired everything); 1 otherwise.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    fix = "--fix" in args
    allow_new = "--allow-new" in args
    unknown = [a for a in args if a not in ("--fix", "--allow-new")]
    if unknown:
        print(f"ERROR: unknown argument(s): {unknown}", file=sys.stderr)
        return 2
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

    # ``--fix``: apply the mechanical bookkeeping and re-report what remains.
    if fix and violations:
        fixed, remaining = [], []
        for vtype, path, msg in violations:
            if vtype == "grew":
                allowlist[path] = count_lines(REPO_ROOT / path)
                fixed.append(f"pin updated: {path} -> {allowlist[path]}")
            elif vtype in ("shrunk", "orphaned"):
                del allowlist[path]
                fixed.append(f"entry removed: {path} ({vtype})")
            elif vtype == "new_violation" and allow_new:
                allowlist[path] = count_lines(REPO_ROOT / path)
                fixed.append(f"entry ADDED (--allow-new): {path} at {allowlist[path]}")
            else:
                remaining.append((vtype, path, msg))
        if fixed:
            with open(ALLOWLIST_PATH, "w") as f:
                json.dump(allowlist, f, indent=2)
                f.write("\n")
            print(f"FIXED {len(fixed)} allowlist entr{'y' if len(fixed) == 1 else 'ies'}:")
            for line in fixed:
                print(f"  {line}")
            print("Commit tools/file_size_allowlist.json with your change.\n")
        violations = remaining
        if remaining and not allow_new:
            print(
                "NOT fixed (new >800-line files need a deliberate decision; "
                "re-run with --fix --allow-new to add them):"
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
