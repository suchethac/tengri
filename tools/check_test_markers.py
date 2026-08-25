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
    """Markers in effect from ``pytestmark`` in this block's own body.

    Only the **last** assignment counts. Python rebinds the name, so

        pytestmark = pytest.mark.bounds
        ...
        pytestmark = pytest.mark.skipif(...)

    leaves no taxonomy marker at collection time -- ``pytest -m bounds``
    deselects the whole module. Unioning across assignments modelled
    ``pytestmark`` as accumulating and let eight modules through, four of them
    losing their taxonomy marker to a ``skipif`` and four to a ``unit`` marker
    that is not in the taxonomy at all.

    A module wanting both writes one assignment holding a list.
    """
    value: ast.expr | None = None
    for node in body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        value = node.value  # a later assignment replaces an earlier one

    if value is None:
        return set()

    items = list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return _marker_names(items)


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


def _pytestmark_lines(body: list[ast.stmt]) -> list[int]:
    """Line numbers of ``pytestmark`` assignments directly in this block."""
    return [
        node.lineno
        for node in body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
    ]


def rebound_pytestmark_scopes(source: str) -> list[tuple[str, list[int]]]:
    """Scopes assigning ``pytestmark`` more than once, as (scope name, lines).

    Reading only the last assignment (see ``_pytestmark_names``) reports the
    truth, but it still lets the pattern through: the file keeps a marker that
    looks declared and is not. Ten modules in this tree had it, eight of them
    losing their taxonomy marker outright. Refusing it outright is the fix that
    does not need finding again -- a scope wanting several markers writes one
    assignment holding a list.

    Class bodies are checked as well as the module body. pytest honours a
    class-body ``pytestmark`` and Python rebinds the name there too, so the
    identical defect can sit one level in. None do today; a checker whose
    domain is narrower than the rule it enforces is how the previous two holes
    in this file went unnoticed.
    """
    scopes: list[tuple[str, list[int]]] = []

    def visit(body: list[ast.stmt], scope: str) -> None:
        lines = _pytestmark_lines(body)
        if len(lines) > 1:
            scopes.append((scope, lines))
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{scope}::{node.name}" if scope != "<module>" else node.name)

    visit(ast.parse(source).body, "<module>")
    return scopes


def check_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")

    rebound = rebound_pytestmark_scopes(source)
    if rebound:
        return [
            f"{scope}: pytestmark assigned {len(lines)}x -- "
            f"{', '.join(f'L{n}' for n in lines[:-1])} discarded, L{lines[-1]} wins. "
            f"Use one assignment holding a list."
            for scope, lines in rebound
        ]

    file_markers = collect_file_markers(source) & APPROVED_MARKERS
    if file_markers:
        return []
    return collect_function_violations(source)


def main() -> int:
    violations: list[tuple[Path, list[str]]] = []

    # The rebinding check runs over EVERY test module, not just the four
    # taxonomy-enforced trees. It is a different rule: a second `pytestmark`
    # assignment discards a `skipif` as readily as a taxonomy marker, and that
    # hurts `tests/unit` and `tests/inference` just as much. Scoping it to
    # ENFORCED_DIRS would make the guard's domain narrower than the rule it
    # states -- the shape of both earlier holes in this file.
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        rebound = rebound_pytestmark_scopes(path.read_text(encoding="utf-8"))
        if rebound:
            violations.append(
                (
                    path.relative_to(REPO_ROOT),
                    [
                        f"{scope}: pytestmark assigned {len(lines)}x -- "
                        f"{', '.join(f'L{n}' for n in lines[:-1])} discarded, "
                        f"L{lines[-1]} wins. Use one assignment holding a list."
                        for scope, lines in rebound
                    ],
                )
            )

    rebound_paths = {p for p, _ in violations}
    for enforced in ENFORCED_DIRS:
        if not enforced.exists():
            continue
        for path in sorted(enforced.rglob("test_*.py")):
            if path.relative_to(REPO_ROOT) in rebound_paths:
                continue  # already reported above; the marker read is unreliable
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
