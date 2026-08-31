#!/usr/bin/env python3
"""CI guard: verify every check_*.py is wired or explicitly marked not wired.

Every tools/check_*.py file must either:
1. Appear in at least one .github/workflows/*.yml, OR
2. Have a docstring line matching: CI: not wired — <non-empty reason>

The second allows opt-outs with documented reasons (e.g., backlog too large).

Usage
-----
    python tools/check_guard_wiring.py

Exit code 0 when all guards are wired or properly declared; 1 otherwise.
See #2050.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def extract_module_docstring(filepath: Path) -> str | None:
    """Extract the module-level docstring from a Python file."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            return tree.body[0].value.value
    except Exception:
        pass
    return None


def is_wired_in_workflows(guard_name: str) -> bool:
    """Check if guard_name appears in a non-comment run: line in any workflow file.

    Matches both inline (run: python tools/check_*.py) and block-scalar forms
    (run: | with python tools/check_*.py on a subsequent line).
    Only matches on actual run: lines, not comments naming the guard in explanations.
    This prevents false positives from YAML comments that mention the guard.
    """
    stem = guard_name.replace(".py", "")
    # Match: "run:" followed by the guard name with word boundaries (inline form)
    # e.g., "run: python tools/check_foo.py"
    run_pattern = rf"\brun:.*\b{re.escape(stem)}\b"
    # Match: python command on its own line (block-scalar form)
    # e.g., "  python tools/check_foo.py"
    python_pattern = rf"^\s*(?:-\s*)?python\s+tools/{re.escape(stem)}\.py\b"
    for workflow_file in WORKFLOWS_DIR.glob("*.yml"):
        try:
            content = workflow_file.read_text()
            for line in content.splitlines():
                # Skip comment lines
                if line.lstrip().startswith("#"):
                    continue
                if re.search(run_pattern, line) or re.search(python_pattern, line):
                    return True
        except Exception:
            pass
    return False


def is_explicitly_not_wired(docstring: str | None) -> bool:
    """Check if docstring declares: CI: not wired — <non-empty reason> (em-dash or hyphen)."""
    if not docstring:
        return False
    # Match: "CI: not wired — <reason>" or "CI: not wired - <reason>" where reason is non-empty
    match = re.search(r"CI:\s*not\s+wired\s*[—-]\s*(.+)", docstring)
    return bool(match and match.group(1).strip())


def main() -> int:
    """Check guard wiring. Exit 0 if all guards are wired or declared."""
    tools_dir = ROOT / "tools"
    guard_files = sorted(tools_dir.glob("check_*.py"))

    unwired_and_undeclared = []

    for guard_file in guard_files:
        guard_name = guard_file.name
        docstring = extract_module_docstring(guard_file)

        wired = is_wired_in_workflows(guard_name)
        declared = is_explicitly_not_wired(docstring)

        if not wired and not declared:
            unwired_and_undeclared.append(guard_name)

    if unwired_and_undeclared:
        print(
            "FAILED: the following guard(s) are neither wired into CI nor "
            "explicitly marked not wired:",
            file=sys.stderr,
        )
        for guard_name in unwired_and_undeclared:
            print(f"  {guard_name}", file=sys.stderr)
        print(
            "\nFix: either",
            file=sys.stderr,
        )
        msg1 = "  1. Add a '- run: python tools/<guard>.py' line to .github/workflows/tests.yml"
        print(msg1, file=sys.stderr)
        msg2 = "  2. Add this line to the module docstring: CI: not wired — <reason>"
        print(msg2, file=sys.stderr)
        return 1

    print(f"OK: all {len(guard_files)} guard(s) are wired or declared.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
