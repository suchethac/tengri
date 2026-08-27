# SPDX-License-Identifier: BSD-3-Clause
"""Pin the absolute amplitude of AGN disc reddening.

Regression test for the convention introduced when prevot_smc was
normalized to k(V)=1 (commit 7de0cef). The legacy formula
``10^(-0.4 * k * agn_ebv_disc)`` worked correctly when prevot_smc
returned the unnormalized k(λ) = A(λ)/E(B-V) (so k(V) ≈ 2.475).

After normalization to k(V) = 1, the same formula instead treats
``agn_ebv_disc`` as A(V), not E(B-V). The user-facing parameter is
named ``agn_ebv_disc`` and documented as E(B-V), so the formula
must restore the R_V multiplier to keep the parameter's semantics.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.unified import _redden_disc

# One assignment, not two: Python rebinds the name, so a second
# `pytestmark = ...` silently discarded the taxonomy marker and left this
# module unselectable by `pytest -m bounds`.
pytestmark = [pytest.mark.bounds, pytest.mark.unit]


# Prevot+1984 SMC law: R_V = A(V)/E(B-V) = 2.72
PREVOT_R_V = 2.72


class TestReddenDiscAmplitude:
    """Pin V-band attenuation against the documented E(B-V) semantics."""

    def test_zero_ebv_is_noop(self):
        wave = jnp.linspace(2000.0, 10000.0, 200)
        l_disc = jnp.ones_like(wave)
        out = _redden_disc(wave, l_disc, agn_ebv_disc=0.0)
        np.testing.assert_allclose(np.asarray(out), 1.0, rtol=1e-12)

    def test_v_band_attenuation_uses_R_V(self):
        """At V band (5500 Å), A(V) = R_V * E(B-V), so flux ratio = 10^(-0.4 * R_V * ebv)."""
        wave = jnp.array([5500.0])
        l_disc = jnp.array([1.0])
        ebv = 0.1
        out = float(np.asarray(_redden_disc(wave, l_disc, agn_ebv_disc=ebv))[0])
        expected = 10.0 ** (-0.4 * PREVOT_R_V * ebv)  # ≈ 0.778
        assert out == pytest.approx(expected, rel=1e-3)

    def test_uv_attenuation_stronger_than_v(self):
        """k(λ) is higher in the UV than at V; attenuation must be stronger there."""
        wave = jnp.array([1500.0, 5500.0, 8000.0])
        l_disc = jnp.ones_like(wave)
        out = np.asarray(_redden_disc(wave, l_disc, agn_ebv_disc=0.2))
        assert out[0] < out[1] < out[2], f"expected UV < V < NIR transmission; got {out}"
