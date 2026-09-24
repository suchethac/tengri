# SPDX-License-Identifier: BSD-3-Clause
"""Configuration VI's AGN amplitude must be able to reach zero (#2495).

Seventeen of the twenty galaxies in the locked sample are not AGN candidates,
so the row fit on all of them has to be able to answer "no AGN". The original
``agn_log_lbol ~ Uniform(9.42, 13.42)`` could not: its floor is ~1e43 erg/s,
already a Seyfert. The fits piled against it -- galaxy 267 put 32% of its draws
and 33% of its 72 divergent draws within 0.05 dex of the floor, and galaxy 79
recorded 107 divergences.

``agn_lum_ratio`` reaches zero, where the row nests Configuration I (measured
4.62e-06 relative in photometry on the CANDELS filter set, against 4.4e-06
measured independently in #2495), so "does this galaxy need an AGN?" becomes
one parameter's posterior rather than a wall.

These assertions read the **declaration**, not a built model, following
``check_config_names.py``: building Configuration VI needs an SSP grid, a Cue
table and a SKIRTOR library, and needing data is what lets a bad declaration
hide on the machine that cannot build it. The numerical nesting is a one-off
measurement recorded in ``config_VI``'s docstring; what is guarded here is the
declaration that makes it possible.

Two regressions this catches, both of which reintroduce the wall silently:

1. ``agn_lum_ratio`` pinned again (by ``Fixed`` or by falling back to the
   ``all_params`` wildcard), which removes the zero-reaching amplitude;
2. an ``agn_log_lbol`` prior reinstated, which restores the floored
   parametrization beside the ratio and re-adds a free parameter.

It also pins ``log_lbol`` to an explicit value rather than ``DEFAULT``. The
registry default is 10.0 while the A/B that justified this change ran at 11.0,
and ``agn_lum_ratio`` multiplies that scale -- so taking the wildcard would
shift what every ratio value means by a factor of ten away from the
measurement, with nothing failing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

CONFIGS_PY = Path(__file__).resolve().parents[1] / "configs.py"


def _agn_group() -> dict[str, ast.expr]:
    """The ``agn={...}`` keyword of ``config_VI``, as literal AST nodes."""
    tree = ast.parse(CONFIGS_PY.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "config_VI"
    )
    call = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "build"
    )
    agn = next(kw.value for kw in call.keywords if kw.arg == "agn")
    return {k.value: v for k, v in zip(agn.keys, agn.values) if isinstance(k, ast.Constant)}


def _call_name(node: ast.expr) -> str | None:
    return getattr(node.func, "id", None) if isinstance(node, ast.Call) else None


def test_the_agn_amplitude_is_a_free_ratio_that_reaches_zero():
    group = _agn_group()
    assert "lum_ratio" in group, (
        "Configuration VI declares no agn_lum_ratio; the AGN amplitude is pinned "
        "by the all_params wildcard and the row cannot express 'no AGN'"
    )
    node = group["lum_ratio"]
    assert _call_name(node) == "Uniform", (
        f"agn_lum_ratio must be free to reach zero, got {ast.unparse(node)}"
    )
    lo = node.args[0].value
    assert lo == 0.0, f"agn_lum_ratio's lower bound must be exactly 0.0, got {lo}"


def test_no_floored_log_lbol_prior_is_reinstated_beside_the_ratio():
    node = _agn_group().get("log_lbol")
    assert node is not None, "log_lbol should stay explicitly pinned, not left to the wildcard"
    assert _call_name(node) != "Uniform", (
        f"a free agn_log_lbol restores the ~1e43 erg/s floor this row was changed "
        f"to escape (#2495): {ast.unparse(node)}"
    )


def test_log_lbol_is_pinned_at_the_value_the_ab_measured():
    node = _agn_group()["log_lbol"]
    assert _call_name(node) == "Fixed", f"expected Fixed(...), got {ast.unparse(node)}"
    (arg,) = node.args
    assert isinstance(arg, ast.Constant), (
        "log_lbol must be pinned at an explicit number: the registry default is "
        "10.0, the A/B ran at 11.0, and agn_lum_ratio multiplies this scale"
    )
    assert arg.value == 11.0, f"A/B ran at log_lbol = 11.0, declaration says {arg.value}"
