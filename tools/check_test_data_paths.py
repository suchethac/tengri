#!/usr/bin/env python3
"""CI guard: a ``Path(__file__)``-rooted path in tests/ must land inside the repo.

Counting ``parents[N]`` by hand has now gone wrong four separate times, in
eight files, and every instance failed the same silent way::

    _DATA = Path(__file__).resolve().parents[4] / "data"  # one too far
    _DATA = Path(__file__).resolve().parents[2] / "data"  # one too few
    _DATA = Path(__file__).parent.parent.parent.parent / "data"

Nothing raises. ``Path`` composes any number of ``..`` steps happily, the
resulting directory simply holds no grids, and the guard built on it --
invariably ``if not (_DATA / grid).exists(): pytest.skip(...)`` -- becomes
permanently true. A skip that can never be lifted is indistinguishable from a
passing test in every report CI produces, so the tests stay dark for as long
as nobody counts them.

The tally when this guard was written: 27 tests never ran. Among them were
gradient tests for Cue and MAPPINGS whose grids are *tracked in git* and
present on every runner, and a Silva04 collapse test that fails a real
assertion the moment it is allowed to execute.

Two rules, both mechanical
--------------------------

1. The directory a ``Path(__file__)`` chain resolves to must be the repository
   root or below it. Anything above is out of bounds -- it exists on the
   author's machine (``.../worktrees``, ``$HOME``) which is exactly why it
   never raised.

2. When that chain is joined to string literals, the **first** joined component
   must exist. That is the component which tests the anchor, and it is as far
   as this can go: deeper components are legitimately absent (an optional grid
   nobody installed) and can be directories as readily as files
   (``data/synthesizer_grids/``), so requiring those to exist would flag a
   dozen healthy sites and make the guard unusable -- the same outcome as not
   having one.

Rule 2 is what catches ``parents[2] / "data"`` from ``tests/physics/gradients``:
``tests/data`` is inside the repository, so rule 1 is satisfied, and the
directory has never existed.

There is deliberately no allowlist. Both rules describe a path that cannot be
correct rather than a style preference, so an exemption would only ever record
a bug someone chose not to fix.

Usage
-----
    python tools/check_test_data_paths.py

Exit code 0 when every expression resolves inside the repository onto a
directory that exists; 1 otherwise, listing file, line and resolved target.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def _strip_calls(node: ast.expr) -> ast.expr:
    """Unwrap ``.resolve()`` / ``.absolute()`` wrappers."""
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("resolve", "absolute")
        and not node.args
    ):
        node = node.func.value
    return node


def _is_path_dunder_file(node: ast.expr) -> bool:
    """True for ``Path(__file__)``, with or without ``.resolve()``."""
    node = _strip_calls(node)
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return False
    if node.func.id != "Path":
        return False
    return any(isinstance(a, ast.Name) and a.id == "__file__" for a in node.args)


def _climb(node: ast.expr) -> tuple[int, ast.expr] | None:
    """Count how many levels up a ``.parent`` / ``.parents[N]`` chain walks.

    ``p.parent`` and ``p.parents[0]`` are the same directory, so ``parents[N]``
    contributes ``N + 1`` levels and each ``.parent`` contributes one. Returns
    ``(levels, base_expression)`` or None when this is not such a chain.
    """
    levels = 0
    while True:
        node = _strip_calls(node)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr != "parents":
                break
            index = node.slice
            if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
                return None  # a computed index -- cannot be resolved statically
            levels += index.value + 1
            node = node.value.value
        elif isinstance(node, ast.Attribute) and node.attr == "parent":
            levels += 1
            node = node.value
        else:
            break
    return (levels, node) if levels else None


def _literal_segments(node: ast.expr) -> tuple[ast.expr, list[str]] | None:
    """Flatten ``base / "a" / "b"`` into ``(base, ["a", "b"])``.

    Returns None when any right-hand operand is not a string literal, so a
    path built from a variable is checked by rule 1 only.
    """
    segments: list[str] = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
            return None
        segments.insert(0, node.right.value)
        node = node.left
    return (node, segments) if segments else None


def _check_file(path: Path) -> list[str]:
    """Every rule violation in one test module."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    rel = path.relative_to(REPO_ROOT)
    problems: list[str] = []
    seen: set[tuple[int, int]] = set()

    for node in ast.walk(tree):
        flat = _literal_segments(node)
        chain_node, segments = flat if flat else (node, [])

        climbed = _climb(chain_node)
        if not climbed:
            continue
        levels, base = climbed
        if not _is_path_dunder_file(base):
            continue

        key = (node.lineno, node.col_offset)
        if key in seen:
            continue
        seen.add(key)

        try:
            anchor = path.resolve().parents[levels - 1]
        except IndexError:
            problems.append(f"{rel}:{node.lineno}: walks {levels} levels up, past the filesystem")
            continue

        if anchor != REPO_ROOT and REPO_ROOT not in anchor.parents:
            problems.append(
                f"{rel}:{node.lineno}: {levels} levels up resolves to {anchor}, outside "
                f"the repository -- {levels - _depth(path)} level(s) too far (the root "
                f"is {_depth(path)} up from here)"
            )
            continue

        if not segments:
            continue
        # Check only the FIRST joined component. That is the part which tests
        # the anchor: `parents[2] / "data"` from tests/physics/gradients builds
        # `tests/data`, and no checkout has ever had one. Deeper components are
        # left alone because an optional bundle is legitimately absent, and it
        # can be a directory (`data/synthesizer_grids/`) just as easily as a
        # file -- flagging those would make this guard unusable, which is the
        # same failure as not having it.
        head = anchor / segments[0]
        if head.exists():
            continue
        problems.append(
            f"{rel}:{node.lineno}: {levels} levels up is {_show(anchor) or '<repo root>'}, "
            f"so this builds {_show(head)}{'/...' if len(segments) > 1 else ''} -- "
            f"no such directory"
        )

    return problems


def _depth(path: Path) -> int:
    """Levels from ``path`` up to the repository root."""
    return len(path.resolve().relative_to(REPO_ROOT).parts)


def _show(target: Path) -> str:
    try:
        return str(target.relative_to(REPO_ROOT))
    except ValueError:
        return str(target)


def main() -> int:
    problems: list[str] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        problems.extend(_check_file(path))

    if problems:
        print("Path(__file__) expressions that cannot resolve as intended:\n")
        for line in problems:
            print(f"  {line}")
        print(
            f"\n{len(problems)} problem(s). Import the shared markers from "
            f"tests/_data_skip.py rather than recomputing a repository root."
        )
        return 1

    print("OK -- every Path(__file__) chain under tests/ lands inside the repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
