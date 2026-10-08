#!/usr/bin/env python3
"""CI guard: no dtype-free ``jnp.asarray`` on a module-level numpy constant in src/tengri/.

``jnp.asarray(NP_CONST)`` with no ``dtype`` takes its dtype from the x64 state at the
moment it runs. When the first call happens under ``jax_enable_x64=False``, the
float32 buffer is cached against the numpy object and later served back as float64,
and the first jitted op on it raises ``RuntimeProgramInputMismatch`` (#2774).
``tengri.utils.host_array.device_table`` passes the canonical dtype explicitly.

A module-level name is a numpy constant when it is bound (directly, by tuple
unpacking, or by aliasing) to a call rooted at ``np``/``numpy`` (including names
imported from a numpy module), to ``host_array(...)``, or to a same-module function
whose body calls into numpy. The set of constant names is collected across all of
src/tengri so that a constant imported from another module is recognized at its use.

Usage
-----
    python tools/check_jnp_asarray_module_constants.py

Exit code 0 when no offending site is found; 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUMPY_MODULES = ("numpy",)
HOST_HELPERS = ("host_array",)
MESSAGE = "jnp.asarray without dtype on module-level numpy constant"


def _root_name(node: ast.expr) -> str | None:
    """Return the leftmost Name of an attribute/call/subscript chain, if any."""
    while True:
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None


def _numpy_roots(tree: ast.Module) -> set[str]:
    """Local names that denote numpy in this file: ``np``, ``numpy``, and from-imports."""
    roots = {"np", "numpy"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in NUMPY_MODULES
        ):
            roots.update(alias.asname or alias.name for alias in node.names)
    return roots


def _calls_numpy(node: ast.AST, roots: set[str]) -> bool:
    """True if any call inside ``node`` is rooted at numpy or is a host-array helper."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        if _root_name(sub.func) in roots:
            return True
        if isinstance(sub.func, ast.Name) and sub.func.id in HOST_HELPERS:
            return True
    return False


def _is_numpy_value(value: ast.expr, np_funcs: set[str], roots: set[str], names: set[str]) -> bool:
    """True if a module-level right-hand side produces a numpy array."""
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name) and func.id in (*HOST_HELPERS, *np_funcs):
            return True
        return _root_name(func) in roots
    if isinstance(value, (ast.Subscript, ast.Attribute, ast.Name)):
        return _root_name(value) in names
    return False


def module_numpy_constants(tree: ast.Module) -> set[str]:
    """Names bound at module scope to numpy arrays (see module docstring)."""
    roots = _numpy_roots(tree)
    np_funcs = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and _calls_numpy(node, roots)
    }
    names: set[str] = set()
    # Two passes resolve aliases of constants defined later in the file.
    for _ in range(2):
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not _is_numpy_value(value, np_funcs, roots, names):
                continue
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


def _locally_bound(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names bound inside a function (parameters and assignment targets)."""
    bound = {arg.arg for arg in [*func.args.args, *func.args.kwonlyargs]}
    for sub in ast.walk(func):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            bound.add(sub.id)
    return bound


class AsarrayConstantVisitor(ast.NodeVisitor):
    """Flags ``jnp.asarray(X)`` with no dtype where X is a module-level numpy constant."""

    def __init__(self, np_names: set[str]):
        self.np_names = np_names
        self.violations: list[tuple[int, str]] = []
        self._shadowed: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track names rebound locally so they do not count as module constants."""
        outer = self._shadowed
        self._shadowed = outer | _locally_bound(node)
        self.generic_visit(node)
        self._shadowed = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        """Flag a dtype-free ``jnp.asarray`` of a numpy constant."""
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "asarray"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "jnp"
            and len(node.args) == 1
            and not any(kw.arg == "dtype" for kw in node.keywords)
            and self._names_constant(node.args[0])
        ):
            self.violations.append((node.lineno, MESSAGE))
        self.generic_visit(node)

    def _names_constant(self, arg: ast.expr) -> bool:
        base = arg.value if isinstance(arg, ast.Subscript) else arg
        if not isinstance(base, ast.Name):
            return False
        return base.id in self.np_names and base.id not in self._shadowed


def parse_source(filepath: Path) -> ast.Module | None:
    """Parse one file; a file that does not parse is skipped with a warning."""
    try:
        return ast.parse(filepath.read_text(), filename=str(filepath))
    except SyntaxError:
        print(f"Warning: {filepath} does not parse, skipping", file=sys.stderr)
        return None


def check_tree(tree: ast.Module, np_names: set[str]) -> list[tuple[int, str]]:
    """Return (lineno, message) for every offending site in one parsed module."""
    visitor = AsarrayConstantVisitor(np_names)
    visitor.visit(tree)
    return visitor.violations


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Return (lineno, message) for offending sites in one file, using its own constants."""
    tree = parse_source(filepath)
    if tree is None:
        return []
    return check_tree(tree, module_numpy_constants(tree))


def _tracked_python_files_in_src() -> list[Path]:
    """Tracked .py files under src/tengri/."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/tengri/*.py", "src/tengri/**/*.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [ROOT / name for name in out.decode("utf-8").split("\0") if name]


def main(argv: Sequence[str] | None = None) -> int:
    """Scan src/tengri for dtype-free jnp.asarray calls on numpy constants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    trees: dict[Path, ast.Module] = {}
    for filepath in _tracked_python_files_in_src():
        tree = parse_source(filepath)
        if tree is not None:
            trees[filepath] = tree
    np_names: set[str] = set()
    for tree in trees.values():
        np_names |= module_numpy_constants(tree)
    offenders: list[str] = []
    for filepath, tree in trees.items():
        rel = filepath.relative_to(ROOT)
        offenders.extend(
            f"  {rel}:{lineno}: {message}" for lineno, message in check_tree(tree, np_names)
        )
    if offenders:
        print("FAILED: dtype-free jnp.asarray on module-level numpy constant(s):", file=sys.stderr)
        print("\n".join(offenders), file=sys.stderr)
        print(
            "Use tengri.utils.host_array.device_table(CONST) at the point of use.",
            file=sys.stderr,
        )
        return 1
    print("OK: no dtype-free jnp.asarray on module-level numpy constants", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
