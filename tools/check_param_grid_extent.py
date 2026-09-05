#!/usr/bin/env python3
"""CI guard: declared grid-backed AGN priors equal their vendored grid extent.

Mechanism M-A ("declared priors/defaults disagree with the data they gate",
Task 1 of the AGNfitter-rX parity plan): a grid-backed parameter's declared
``Uniform(lo, hi)`` is supposed to be the range the vendored HDF5 template
grid can actually serve, but nothing enforced that. ``agn_log_nh_silva``
declared ``[22, 25]`` while ``data/silva04_torus_grid.h5`` spanned
``[21.5, 24.45]`` for a long time; ``agn_oa_skirtor`` declared ``[20, 60]``
while every vendored SKIRTOR grid (``skirtor_mean3p_torus_grid.h5``,
``skirtor_templates_v2.h5``, ``skirtor_templates_v3.h5``,
``agnfitter_torus_reference.h5``) spans ``[10, 80]``. Both silently clip a fit
that samples the declared edge onto whatever the grid's real edge is, with a
zero gradient in the gap and nothing to reveal it (#1586's failure mode,
applied to the *declaration* rather than a downstream consumer).

This guard reads the module-level ``GRID_EXTENT_SOURCES`` registry in
``src/tengri/components/agn/_params.py`` and, for each entry, opens the named
HDF5 file, reads the named axis, and asserts the declared ``PARAMS`` bound for
that entry's parameter equals the axis extent (``axis.min()``/``axis.max()``,
optionally transformed) to within an absolute tolerance of 1e-9.

Deliberately excluded parameters (a real grid-backed prior with no matching
entry here) are not oversights: see the ``GRID_EXTENT_SOURCES`` docstring in
``_params.py`` for the specific reasoning per exclusion (mostly: the
parameter is *shared* across several blocks with different native grids, so
pinning the shared declaration to any one of them would be wrong, and the
existing ``components/grid_support.py`` mechanism already reconciles that
case per-block, #1586).

Why this parses ``_params.py`` with ``ast`` rather than importing it
----------------------------------------------------------------------
Importing ``tengri.components.agn._params`` transitively imports the
``tengri`` package (Python always runs a package's ``__init__.py`` before any
submodule), which pulls in JAX and the rest of the import-heavy stack. This
guard only needs two literal, static structures out of one module -- the
``ParamDeclaration(name, Uniform(lo, hi, ...), ...)`` calls inside ``PARAMS``
and the ``GRID_EXTENT_SOURCES`` dict literal -- so it reads the source text
with :mod:`ast` instead. This keeps the guard runnable with only ``h5py`` and
``numpy`` installed (no editable ``tengri`` install required), and immune to
any *unrelated* import-time failure elsewhere in the package.

Usage
-----
    python tools/check_param_grid_extent.py

Exit code 0 if every registered entry matches its vendored grid extent; 1
with a table of mismatches otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = ROOT / "src" / "tengri" / "components" / "agn" / "_params.py"

#: Absolute tolerance on the lo/hi comparison. A prior transcribed from a grid
#: axis matches it exactly; the one case that does not (an inclination axis
#: run through cos(deg2rad(...))) lands within a few ULP of an exact endpoint
#: (e.g. cos(90 deg) = 6.123e-17, not bit-exact 0.0), which this tolerance
#: absorbs without hiding an actual mismatch (real mismatches here are always
#: at least 0.5, three orders of magnitude above this).
TOL = 1e-9

Bounds = tuple[float, float]
GridSource = tuple[str, str, str, str]  # (param_name, h5_path, dataset_path, transform)


def _parse_params_module() -> ast.Module:
    """Parse ``_params.py`` into an AST without importing it (or ``tengri``)."""
    return ast.parse(PARAMS_FILE.read_text(), filename=str(PARAMS_FILE))


def _extract_declared_bounds(tree: ast.Module) -> dict[str, Bounds]:
    """Return ``{param_name: (lo, hi)}`` for every literal ``Uniform(lo, hi, ...)``
    prior inside a ``ParamDeclaration(name, Uniform(lo, hi, ...), ...)`` call in
    ``tree``.

    Non-literal priors (``LogUniform``, ``Fixed``, or a ``Uniform`` built from
    a computed expression) are silently skipped: no entry in
    ``GRID_EXTENT_SOURCES`` should ever name one, and if it does,
    ``_check_one`` below reports it as "NOT DECLARED" rather than crashing.
    """
    bounds: dict[str, Bounds] = {}
    for node in ast.walk(tree):
        is_param_declaration_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ParamDeclaration"
            and len(node.args) >= 2
        )
        if not is_param_declaration_call:
            continue
        name_node, prior_node = node.args[0], node.args[1]
        if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
            continue
        is_uniform_call = (
            isinstance(prior_node, ast.Call)
            and isinstance(prior_node.func, ast.Name)
            and prior_node.func.id == "Uniform"
            and len(prior_node.args) >= 2
        )
        if not is_uniform_call:
            continue
        try:
            lo = float(ast.literal_eval(prior_node.args[0]))
            hi = float(ast.literal_eval(prior_node.args[1]))
        except (ValueError, TypeError, SyntaxError):
            continue
        bounds[name_node.value] = (lo, hi)
    return bounds


def _extract_grid_extent_sources(tree: ast.Module) -> dict[str, GridSource]:
    """Return the literal ``GRID_EXTENT_SOURCES`` dict from ``tree``.

    Raises
    ------
    RuntimeError
        If ``GRID_EXTENT_SOURCES`` is not assigned at module level in
        ``_params.py``, or is not a fully literal dict (``ast.literal_eval``
        fails) -- either means the registry contract described in its own
        docstring was broken, which this guard cannot silently work around.
    """
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "GRID_EXTENT_SOURCES":
            return ast.literal_eval(value)
    raise RuntimeError(f"GRID_EXTENT_SOURCES not found in {PARAMS_FILE}")


def _axis_extent(h5_path: Path, dataset: str, transform: str) -> Bounds:
    """Read one HDF5 dataset and return its (optionally transformed) extent."""
    with h5py.File(h5_path, "r") as f:
        axis = np.asarray(f[dataset][:], dtype=np.float64)
    if transform == "cos_deg":
        axis = np.cos(np.deg2rad(axis))
    elif transform != "identity":
        raise ValueError(f"unknown transform {transform!r} for dataset {dataset!r}")
    return float(axis.min()), float(axis.max())


def _fmt(bounds: Bounds) -> str:
    lo, hi = bounds
    return f"[{lo:.6g}, {hi:.6g}]"


def main() -> int:
    """Compare every ``GRID_EXTENT_SOURCES`` entry against its vendored grid.

    Prints one row per entry and returns 1 if any row mismatches (or its file
    / dataset / declaration is missing), 0 otherwise.
    """
    tree = _parse_params_module()
    declared = _extract_declared_bounds(tree)
    sources = _extract_grid_extent_sources(tree)

    rows: list[tuple[str, str, str, str, str]] = []
    ok = True
    for key in sorted(sources):
        param_name, h5_rel, dataset, transform = sources[key]

        if param_name not in declared:
            rows.append((key, param_name, "NOT DECLARED", "-", "FAIL"))
            ok = False
            continue
        declared_str = _fmt(declared[param_name])

        h5_path = ROOT / h5_rel
        if not h5_path.is_file():
            rows.append((key, param_name, declared_str, f"MISSING FILE {h5_rel}", "FAIL"))
            ok = False
            continue

        try:
            grid_bounds = _axis_extent(h5_path, dataset, transform)
        except KeyError:
            rows.append((key, param_name, declared_str, f"MISSING DATASET {dataset}", "FAIL"))
            ok = False
            continue

        match = all(
            abs(d - g) <= TOL for d, g in zip(declared[param_name], grid_bounds, strict=True)
        )
        status = "PASS" if match else "FAIL"
        ok = ok and match
        rows.append((key, param_name, declared_str, _fmt(grid_bounds), status))

    col = (26, 20, 18, 26)
    header = (
        f"{'block':<{col[0]}} {'param':<{col[1]}} {'declared':<{col[2]}} "
        f"{'grid extent':<{col[3]}} status"
    )
    print(header)
    print("-" * len(header))
    for key, param_name, declared_str, grid_str, status in rows:
        print(
            f"{key:<{col[0]}} {param_name:<{col[1]}} {declared_str:<{col[2]}} "
            f"{grid_str:<{col[3]}} {status}"
        )

    n_fail = sum(1 for row in rows if row[-1] == "FAIL")
    if ok:
        print(f"\nOK: all {len(rows)} grid-backed prior(s) match their vendored extent.")
    else:
        print(
            f"\nFAILED: {n_fail}/{len(rows)} grid-backed prior(s) disagree with "
            "their vendored extent (see table above).",
            file=sys.stderr,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
