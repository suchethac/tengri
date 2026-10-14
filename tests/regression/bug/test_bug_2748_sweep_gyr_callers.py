# SPDX-License-Identifier: BSD-3-Clause
"""No notebook, gallery example or benchmark script calls a refused timing name.

#2748 reparametrized the 21 SFH timing parameters as ``*_frac`` (a fraction of
the age of the universe at the galaxy's redshift). The old ``*_gyr`` names
survive only as Fixed aliases, and three call shapes are now refused:

1. a free prior (``Uniform(...)`` or ``FREE``) on a ``*_gyr`` timing name;
2. a ``*_gyr`` name given together with its ``*_frac`` twin.

Those shapes are caught at build time, so a caller that still uses them fails
only when someone runs it. This test reads the caller sources statically and
names every offending line, so the failure is visible without building a model.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.parameters.timing import timing_frac_params

pytestmark = pytest.mark.regression_bug

REPO = pathlib.Path(__file__).resolve().parents[3]

_CALLER_GLOBS = (
    "notebooks/*.py",
    "examples/**/*.py",
    "bench/scripts/*.py",
)


def _gyr_names() -> frozenset[str]:
    """Every absolute-Gyr timing name, full and as a group-local short key."""
    full = set(timing_frac_params().values())
    short = set()
    for group, spec in SFH_REGISTRY.items():
        for name in spec.params:
            if name in full:
                short.add(name.removeprefix(f"sfh_{group}_"))
    return frozenset(full | short)


def _frac_twin(gyr: str) -> str:
    """The ``*_frac`` name of a ``*_gyr`` name, full or short."""
    return gyr[: -len("_gyr")] + "_frac"


def _caller_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for pattern in _CALLER_GLOBS:
        out.extend(p for p in REPO.glob(pattern) if "archive" not in p.parts)
    return sorted(out)


def _key_name(node: ast.AST) -> str | None:
    """The name a dict key or keyword binds, or None for anything dynamic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_free_prior(value: ast.AST) -> bool:
    """A distribution call or a bare FREE sentinel: a prior, not a Fixed value."""
    if isinstance(value, ast.Name):
        return value.id == "FREE"
    if isinstance(value, ast.Call):
        func = value.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return callee != "Fixed" and callee[:1].isupper()
    return False


def _violations(path: pathlib.Path) -> list[str]:
    gyr = _gyr_names()
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        pairs: list[tuple[str, ast.AST, int]] = []
        if isinstance(node, ast.Dict):
            pairs = [(_key_name(k), v, k.lineno) for k, v in zip(node.keys, node.values) if k]
        elif isinstance(node, ast.Call):
            pairs = [(kw.arg, kw.value, kw.value.lineno) for kw in node.keywords if kw.arg]
        keys = {name for name, _, _ in pairs if name}
        for name, value, line in pairs:
            if name in gyr and _is_free_prior(value):
                found.append(f"{path.relative_to(REPO)}:{line}: free prior on {name!r}")
            if name in gyr and _frac_twin(name) in keys:
                found.append(
                    f"{path.relative_to(REPO)}:{line}: {name!r} given with {_frac_twin(name)!r}"
                )
    return found


def test_no_caller_frees_or_doubles_a_gyr_timing():
    files = _caller_files()
    assert files, "caller globs matched nothing: the scan would pass vacuously"
    found = [v for path in files for v in _violations(path)]
    assert not found, "refused *_gyr timing forms in callers:\n" + "\n".join(found)
