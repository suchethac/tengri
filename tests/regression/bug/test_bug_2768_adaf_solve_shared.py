# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2768: the ADAF's scalar solve is traced once per composition.

The runner evaluates the disc block on the caller's grid, at the 2500/4400 A anchors, at 5100 A
and on the budget grids. Each evaluation re-solved the ADAF (four unrolled ``T_e`` solves,
15 387 jaxpr equations, 49 975 under grad), and a BLR block reads the 5100 A evaluation, so a
composition with the analytic BLR carried several differentiated copies (2.60 x the equations of
one ``adaf_spectrum`` gradient on main, by this test's counter; 72 s cold first gradient). The
runner now solves once (``DISC_STATE_BLOCKS``) and hands the state to every evaluation: 0.98 x,
28.8 s (29 s before #2728).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import adaf as adaf_module
from tengri.components.agn.blocks.runner import compose_l_nu

pytestmark = pytest.mark.regression_bug

_SOLVES_PER_STATE = 4  # three in the mdot fixed point, one for the spectrum
_EQUATION_RATIO = 1.3  # composition / one adaf_spectrum; measured 0.98 here, 2.60 on main
_GRID = np.geomspace(1.0e3, 1.0e8, 1000)


def _compose(x, blr_block):
    return jnp.sum(
        compose_l_nu(
            jnp.asarray(_GRID),
            x[0],
            agn_cos_inc=0.5,
            agn_log_mbh=x[1],
            agn_disc_block="adaf",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_norm="independent",
            agn_nlr_block="none",
            agn_blr_block=blr_block,
            agn_feii_block="none",
        )
    )


def _equations(jaxpr) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        total += 1
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    total += _equations(inner)
    return total


@pytest.mark.parametrize("blr_block", ["none", "analytic"])
def test_adaf_solved_once(monkeypatch, blr_block):
    calls = []
    original = adaf_module._adaf_electron_temperature

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(adaf_module, "_adaf_electron_temperature", counting)
    jax.make_jaxpr(lambda x: _compose(x, blr_block))(jnp.array([11.5, 8.0]))
    assert len(calls) == _SOLVES_PER_STATE, f"{len(calls)} T_e solves traced"


def test_gradient_graph_holds_one_adaf_not_one_per_disc_evaluation():
    """A composition with the analytic BLR differentiates one ADAF, not one per disc read.

    The BLR reads the disc at 5100 A, and the runner also evaluates it on the grid, at the
    anchors and on the budget grids; each used to carry its own differentiated solve. The count
    is taken against a single ``adaf_spectrum`` gradient counted by the same function in the
    same process, so it does not depend on how a JAX release lowers ``jnp`` internals: the
    whole composition (disc, lines, ledger) is 0.98 x one spectrum here and 2.60 x on main.
    """
    x0 = jnp.array([11.5, 8.0])
    wave = jnp.asarray(_GRID)

    def single(x):
        return jnp.sum(adaf_module.adaf_spectrum(wave, agn_log_lbol=x[0], agn_log_mbh=x[1]))

    one = _equations(jax.make_jaxpr(jax.grad(single))(x0).jaxpr)
    composition = _equations(jax.make_jaxpr(jax.grad(lambda x: _compose(x, "analytic")))(x0).jaxpr)
    assert composition <= _EQUATION_RATIO * one, (
        f"{composition:,} vs one spectrum {one:,} ({composition / one:.2f})"
    )


def test_state_sharing_changes_no_number_in_a_composition(monkeypatch):
    """Components, anchors and lines are bit-identical with and without the shared state."""
    from tengri.components.agn.blocks import runner

    x0 = jnp.array([11.5, 8.0])

    def run():
        return compose_l_nu(
            jnp.asarray(_GRID),
            x0[0],
            agn_cos_inc=0.5,
            agn_log_mbh=x0[1],
            agn_disc_block="adaf",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_norm="conserving",
            agn_nlr_block="analytic",
            agn_blr_block="analytic",
            agn_feii_block="boroson_green",
            agn_fe2_strength=1.0,
            return_components=True,
        )

    shared = jax.tree_util.tree_leaves(run())
    monkeypatch.setattr(runner, "DISC_STATE_BLOCKS", {})
    unshared = jax.tree_util.tree_leaves(run())
    assert len(shared) == len(unshared) > 1
    for a, b in zip(shared, unshared, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=0.0)
