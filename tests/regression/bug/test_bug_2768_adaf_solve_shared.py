# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2768: the ADAF's scalar solve is traced once per composition.

The runner evaluates the disc block on the caller's grid, at the 2500/4400 A anchors, at 5100 A
and on the budget grids. Each evaluation re-solved the ADAF (four unrolled ``T_e`` solves,
15 387 jaxpr equations, 49 975 under grad), and a BLR block reads the 5100 A evaluation, so a
composition with the analytic BLR carried two differentiated copies: 101 609 equations, a 72 s
cold first gradient. The runner now solves once (``DISC_STATE_BLOCKS``) and hands the state to
every evaluation: 52 018 equations, 28.8 s (29 s before #2728).
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
_EQUATION_BUDGET = 62_000  # 1.2 x 52 018 measured; 101 609 before
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


def test_gradient_equation_count_with_the_analytic_blr():
    jaxpr = jax.make_jaxpr(jax.grad(lambda x: _compose(x, "analytic")))(jnp.array([11.5, 8.0]))
    n = _equations(jaxpr.jaxpr)
    assert n <= _EQUATION_BUDGET, f"{n:,} > {_EQUATION_BUDGET:,}"


def test_shared_state_reproduces_the_standalone_spectrum():
    wave = jnp.asarray(_GRID)
    kwargs = {"agn_log_mbh": 8.3, "agn_adaf_alpha": 0.2, "agn_adaf_beta": 0.6}
    state = adaf_module.adaf_scalar_state(11.2, 1.0, dtype=wave.dtype, **kwargs)
    shared = adaf_module.adaf_spectrum_from_state(wave, state)
    alone = adaf_module.adaf_spectrum(wave, agn_log_lbol=11.2, **kwargs)
    np.testing.assert_allclose(np.asarray(shared), np.asarray(alone), rtol=0.0, atol=0.0)
