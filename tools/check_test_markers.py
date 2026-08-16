#!/usr/bin/env python3
"""CI guard for the physics-test taxonomy declared in tests/TESTING.md.

Every test file under the physics-bearing trees must declare a marker
from the approved taxonomy, either at the file level
(`pytestmark = pytest.mark.<marker>`) or on every test function.

The guard is intentionally narrow: it only enforces the contract in
trees where tests are *supposed* to be physics-shaped. The flat
`tests/unit/` tree is exempt during the rehome transition and graduates
into the contract as files move under the structured layout.

Usage
-----
    python tools/check_test_markers.py

Exit code 0 on success; non-zero with violations listed otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

# Trees where every test file MUST declare one of the markers below.
# Add a directory here when it has been fully rehomed under the contract.
ENFORCED_DIRS: tuple[Path, ...] = (
    TESTS_DIR / "physics",
    TESTS_DIR / "regression",
    TESTS_DIR / "contract",
    TESTS_DIR / "components",
)

APPROVED_MARKERS: frozenset[str] = frozenset(
    {
        "conservation",
        "bounds",
        "limit",
        "regression_paper",
        "regression_bug",
        "gradient",
        "crossval",
        "contract",
    }
)

# Parsed, not pattern-matched. The regex version this replaces had two defects,
# both found on 2026-08-16 by diffing it against this implementation over the
# enforced trees:
#
#   * ``TEST_DEF_RE`` was anchored at column 0 (``^def\s+test_``), so **every
#     test inside a class was invisible to the guard**. 57 unmarked tests
#     across 7 files were passing CI that way -- 14 in one file alone. That is
#     the guard failing open, which is the worse direction.
#   * the decorator scan walked upward from the ``def`` while each line started
#     with ``@``, so a *multi-line* decorator block terminated the walk at its
#     closing paren and hid every marker above it. A test carrying
#     ``@pytest.mark.gradient`` above a wrapped ``@pytest.mark.xfail(...)`` was
#     reported as unmarked.
#
# ``ast`` answers both correctly and also picks up markers inherited from a
# decorated class or a class-body ``pytestmark``, which pytest honours and the
# regex could not see.


def _marker_names(nodes: list[ast.expr]) -> set[str]:
    """Marker names in ``pytest.mark.<name>`` / ``pytest.mark.<name>(...)`` nodes."""
    found: set[str] = set()
    for node in nodes:
        target = node.func if isinstance(node, ast.Call) else node
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
        ):
            found.add(target.attr)
    return found


def _pytestmark_names(body: list[ast.stmt]) -> set[str]:
    """Markers assigned to ``pytestmark`` anywhere in this block's own body."""
    found: set[str] = set()
    for node in body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        value = node.value
        items = list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]
        found |= _marker_names(items)
    return found


def collect_file_markers(source: str) -> set[str]:
    """Return all markers declared at module level via `pytestmark = ...`."""
    return _pytestmark_names(ast.parse(source).body)


def collect_function_violations(source: str) -> list[str]:
    """Return names of test functions without an approved marker.

    Recurses into classes, so a test method is held to the same contract as a
    module-level one, and honours markers inherited from a decorated class or a
    class-body ``pytestmark``.
    """
    violations: list[str] = []

    def walk(block: list[ast.stmt], inherited: set[str]) -> None:
        for node in block:
            if isinstance(node, ast.ClassDef):
                scope = (
                    inherited | _marker_names(node.decorator_list) | _pytestmark_names(node.body)
                )
                walk(node.body, scope)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("test"):
                    continue
                declared = inherited | _marker_names(node.decorator_list)
                if not declared & APPROVED_MARKERS:
                    violations.append(node.name)

    walk(ast.parse(source).body, set())
    return violations


def check_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    file_markers = collect_file_markers(source) & APPROVED_MARKERS
    if file_markers:
        return []
    return collect_function_violations(source)


def main() -> int:
    violations: list[tuple[Path, list[str]]] = []
    for enforced in ENFORCED_DIRS:
        if not enforced.exists():
            continue
        for path in sorted(enforced.rglob("test_*.py")):
            offenders = check_file(path)
            if offenders:
                violations.append((path.relative_to(REPO_ROOT), offenders))

    if not violations:
        print("OK: every enforced test declares a taxonomy marker")
        return 0

    print("FAIL: tests missing taxonomy markers (see tests/TESTING.md)\n")
    for path, offenders in violations:
        print(f"  {path}")
        for fn in offenders:
            print(f"      - {fn}")
    print(
        "\nFix: add `pytestmark = pytest.mark.<marker>` at module top, "
        "or decorate each test with one of: " + ", ".join(sorted(APPROVED_MARKERS))
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
