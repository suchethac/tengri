#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a name may not be defined twice in the same class or module body.

Python binds the second ``def`` and silently discards the first. Editing the
discarded copy changes nothing, and the test suite keeps passing on the
surviving one. ``SEDModel._additive_term_shape_table`` sat in ``sed_model.py``
twice for that reason: #2819 and #2824 each added a copy, and the file had
already carried a twice-defined function once before (#1401).

What this guard catches
-----------------------
Two ``def``, ``async def`` or ``class`` statements that are direct children of
the same scope body, binding the same name. The scopes checked are the module
body and the body of every class.

Why ruff does not catch it
--------------------------
Pyflakes reports this as F811, but only when the first binding is still unused
when the second one arrives. A duplicate method on a class is almost never used
between the two definitions, so the rule passes on ``sed_model.py`` (verified).
This guard looks at the scope as a whole instead.

What it deliberately ignores
----------------------------
* Definitions nested under ``if``, ``try``, ``with`` or ``for``. A conditional
  redefinition (``if TYPE_CHECKING:`` / ``else:``) is legitimate, and those
  statements are not direct children of the body, so they are never compared.
* ``@overload`` stubs, which exist to be redefined.
* Property families: ``@x.setter``, ``@x.getter`` and ``@x.deleter``.
* ``functools.singledispatch`` registrations, ``@f.register`` or
  ``@f.register(int)``.

Dependencies: standard library only, so it runs in the ``lint`` job.

Usage
-----
    python tools/check_duplicate_defs.py
    python tools/check_duplicate_defs.py --list    # print the findings, do not fail

Exit code 0 when no scope binds a name twice; 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Trees scanned. ``src/tengri`` is the package; ``tools`` and ``tests`` are
#: included because a duplicate there hides a broken gate just as well.
SCAN_ROOTS = ("src/tengri", "tools", "tests")

#: Decorator attribute names that mark a deliberate redefinition.
_PROPERTY_ATTRS = frozenset({"setter", "getter", "deleter"})

#: Decorators whose target name is bound again on purpose.
_REDEFINING_NAMES = frozenset({"overload", "register"})

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _decorator_name(node: ast.expr) -> str:
    """Return the trailing name of a decorator, unwrapping a call such as ``register(int)``."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_redefinition(node: ast.stmt) -> bool:
    """Return True for an overload, a property accessor or a singledispatch registration."""
    names = {_decorator_name(dec) for dec in getattr(node, "decorator_list", [])}
    return bool(names & (_PROPERTY_ATTRS | _REDEFINING_NAMES))


def _scopes(tree: ast.Module) -> Iterator[tuple[str, list[ast.stmt]]]:
    """Yield ``(scope name, body)`` for the module and for every class in it."""
    yield "module", tree.body
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node.name, node.body


def _label(path: Path) -> str:
    """Return *path* relative to the repository root when it lies inside it."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _duplicates(path: Path) -> list[str]:
    """Return one finding per name bound twice within a single scope of *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = _label(path)
    findings: list[str] = []
    for scope, body in _scopes(tree):
        first: dict[str, int] = {}
        for stmt in body:
            if not isinstance(stmt, _DEFINITIONS) or _is_redefinition(stmt):
                continue
            if stmt.name in first:
                findings.append(
                    f"{rel}:{stmt.lineno}: duplicate definition of {stmt.name} "
                    f"in {scope} (first at line {first[stmt.name]})"
                )
            else:
                first[stmt.name] = stmt.lineno
    return findings


def _python_files(roots: Sequence[str]) -> list[Path]:
    """Return every ``.py`` file under *roots*, relative to the repository root."""
    return sorted(path for root in roots for path in (REPO_ROOT / root).rglob("*.py"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print the findings without failing")
    args = parser.parse_args(argv)

    findings = [line for path in _python_files(SCAN_ROOTS) for line in _duplicates(path)]

    if args.list:
        print("\n".join(findings))
        return 0

    if not findings:
        print("check_duplicate_defs: OK -- no scope binds a name twice.")
        return 0

    for line in findings:
        print(line, file=sys.stderr)
    print(
        f"\n{len(findings)} duplicate definition(s). Python keeps the last one; the "
        "others are dead code that looks live. Delete the one that is not used.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
