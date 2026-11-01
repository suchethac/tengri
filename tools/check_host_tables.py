#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a module-scope host table reaches JAX only through ``device_table``.

``src/tengri`` keeps its numeric tables (attenuation-law coefficients, IGM
absorber tables, line wavelength lists, ...) as plain numpy arrays at module
scope, built with ``tengri.utils.host_array.host_array``. That is the fix for
#2271: a ``jnp`` array at module scope is a device buffer allocated at import,
so its dtype is fixed by the x64 state at import time, and a backend with no
float64 (jax-mps) cannot even import the package.

A plain numpy table is only half of the fix. JAX 0.11 memoizes a numpy object's
device conversion **by object identity, with the dtype of its first
conversion**: the same module-level array reused across traces under a
different x64 state comes back with the old dtype. Loudly when it meets a
tracer in an operator (``MLIRError: Verification failed`` at
``broadcast_in_dim``, measured on the AGN seam of the precision tree), and
**silently** from ``jnp.take`` / indexing (float64 returned under
``jax.enable_x64(False)``). An explicit dtype at the conversion breaks the
memo, and no array-like wrapper can supply it implicitly: JAX refuses
``__jax_array__`` for jit arguments.

So every use of a table inside a function body must state its intent::

    device_table(_SMC_LAM)  # device copy in this trace's dtype
    np.asarray(_WAVE_REST)  # host-side numpy, stated
    len(_LINE_WAVES), _TABLE.shape  # host metadata

and a bare ``_SMC_LAM`` in an expression, a subscript ``_A_LAF[:, 0:1]``, or
an argument ``jnp.interp(z, _NODES, _VALUES)`` is a violation. Module scope
itself is unconstrained (deriving one table from another there is host
arithmetic). Static, because the hazard depends on the *history of the
process*, which no single test observes: the runtime half lives in
``tests/regression/precision/test_no_float64_constants_under_f32.py``.

Usage::

    python tools/check_host_tables.py            # exit 1 on any violation
    python tools/check_host_tables.py --list     # every table use with its verdict
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "tengri"
HOST_CALLS = {"len", "float", "int", "tuple", "list", "device_table", "host_array"}
HOST_ATTRS = {"shape", "size", "dtype", "ndim"}
#: Names that hold a cached template library (a numpy-backed bundle) in the AGN modules.
LIBRARY_NAMES = {"grid", "aniso", "templates", "template", "library", "bundle", "table"}
LIBRARY_DIR = SRC / "components" / "agn"


def _table_names(tree: ast.Module) -> set[str]:
    """Module-scope names bound to host_array(...) or derived from one at module scope."""
    names: set[str] = set()

    def bound_by_host_array(value: ast.AST) -> bool:
        return "host_array" in ast.dump(value)

    for node in tree.body:
        stmts = [node] + (node.body if isinstance(node, ast.Try) else [])
        for stmt in stmts:
            if not isinstance(stmt, ast.Assign):
                continue
            targets: list[ast.AST] = []
            for t in stmt.targets:
                targets.extend(t.elts if isinstance(t, ast.Tuple) else [t])
            derived = isinstance(stmt.value, (ast.Subscript, ast.BinOp)) and any(
                isinstance(n, ast.Name) and n.id in names for n in ast.walk(stmt.value)
            )
            if bound_by_host_array(stmt.value) or derived:
                names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _verdict(parents: dict[ast.AST, ast.AST], node: ast.Name) -> str:
    child: ast.AST = node
    p = parents.get(node)
    while isinstance(p, ast.Subscript) and p.value is child:
        child, p = p, parents.get(p)
    if isinstance(p, ast.Call) and child in p.args:
        f = p.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "np":
            return "host: np.*"
        if isinstance(f, ast.Name) and f.id == "device_table":
            return "device: device_table"
        if isinstance(f, ast.Name) and f.id in HOST_CALLS:
            return f"host: {f.id}()"
        callee = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "?")
        return f"VIOLATION: bare argument to {callee}()"
    if isinstance(p, ast.Attribute) and p.attr in HOST_ATTRS:
        return f"host: .{p.attr}"
    if isinstance(p, ast.arguments):
        return "host: default value"
    if isinstance(p, ast.Subscript):
        return "VIOLATION: bare subscript"
    return f"VIOLATION: bare in {type(p).__name__}"


def scan(paths: Iterable[Path]) -> list[tuple[Path, int, str, str]]:
    rows: list[tuple[Path, int, str, str]] = []
    for path in paths:
        text = path.read_text()
        if "host_array" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        names = _table_names(tree)
        if not names:
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        in_func: set[ast.AST] = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                in_func.update(ast.walk(fn))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and node.id in names
                and isinstance(node.ctx, ast.Load)
                and node in in_func
            ):
                rows.append((path, node.lineno, node.id, _verdict(parents, node)))
    return rows


def _is_library_read(arg: ast.AST, loop_names: frozenset[str] = frozenset()) -> bool:
    """True when ``arg`` is a field of a template library, a subscript of one, or its
    scale-safe copy; ``loop_names`` are comprehension variables iterating over such fields."""
    if isinstance(arg, ast.Subscript):
        return _is_library_read(arg.value, loop_names)
    if isinstance(arg, ast.Attribute):
        return isinstance(arg.value, ast.Name) and arg.value.id in LIBRARY_NAMES
    if isinstance(arg, ast.Name):
        return arg.id in loop_names
    return isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "float32_safe_template"


def _loop_names(tree: ast.AST) -> dict[ast.AST, frozenset[str]]:
    """For each comprehension, the variables it binds from a library field."""
    out: dict[ast.AST, frozenset[str]] = {}
    for comp in ast.walk(tree):
        if isinstance(comp, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            out[comp] = frozenset(
                g.target.id
                for g in comp.generators
                if isinstance(g.target, ast.Name) and _is_library_read(g.iter)
            )
    return out


def scan_library_conversions(paths: Iterable[Path]) -> list[tuple[Path, int, str, str]]:
    """Bare ``jnp.asarray(library.field)`` conversions in the AGN modules.

    A cached library holds numpy arrays, so a conversion without a stated dtype takes the
    dtype of the first conversion anywhere in the process (see the module docstring). The
    dtype must be stated: ``device_table(library.field)`` or ``jnp.asarray(x, dtype=...)``.
    """
    rows: list[tuple[Path, int, str, str]] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        loops = _loop_names(tree)
        inside: dict[ast.AST, frozenset[str]] = {}
        for comp, names in loops.items():
            for child in ast.walk(comp):
                inside[child] = names
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"asarray", "array"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "jnp"
                and len(node.args) == 1
                and not node.keywords
            ):
                continue
            if _is_library_read(node.args[0], inside.get(node, frozenset())):
                rows.append(
                    (path, node.lineno, ast.unparse(node)[:60], "VIOLATION: bare library read")
                )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--list", action="store_true", help="print every table use with its verdict"
    )
    args = parser.parse_args(argv)
    rows = scan(sorted(SRC.rglob("*.py")))
    library_rows = scan_library_conversions(sorted(LIBRARY_DIR.rglob("*.py")))
    if args.list:
        for path, line, name, verdict in rows + library_rows:
            print(f"{path.relative_to(REPO)}:{line}: {name:<30} {verdict}")
        print(f"{len(rows)} table uses inside function bodies")
        return 0
    bad = [r for r in rows if r[3].startswith("VIOLATION")] + library_rows
    if not rows:
        print("check_host_tables: no host tables found under src/tengri (the census is empty)")
        return 1
    if not bad:
        print(
            f"check_host_tables: {len(rows)} table uses, all stated (device_table / np.* / host)"
        )
        return 0
    print(f"check_host_tables: {len(bad)} table use(s) reach JAX as a bare module-scope array:")
    for path, line, name, verdict in bad:
        print(f"  {path.relative_to(REPO)}:{line}: {name}  {verdict}")
    print(
        "Wrap as device_table(NAME) for the device copy in this trace's dtype, "
        "or np.asarray(NAME) for host use."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
