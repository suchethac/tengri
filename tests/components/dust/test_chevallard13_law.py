# SPDX-License-Identifier: BSD-3-Clause
"""Chevallard et al. (2013) optical-depth-dependent diffuse-ISM attenuation law.

Checks the law against the published equations (Chevallard+2013 eqs 9-11,
restated as eqs 2.11-2.14 of Chevallard & Charlot 2016):

- k(5500 A, tau) = 1 for every tau (V-band normalization);
- the local slope at V equals -n_V(tau) = -2.8 / (1 + 3 sqrt(tau));
- a numpy transcription of the equations agrees on a grid;
- the slope flattens as tau grows;
- the index is held at its 0.1 and 2.5 um values outside the fitted range;
- the gradient with respect to tau_v is finite at tau = 0 and tau = 0.5;
- the law is JIT-compatible and registered as ``chevallard13``.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import chevallard13
from tengri.components.dust.laws._registry import DUST_LAWS
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.regression_paper

_V_BAND_ANGSTROM = 5500.0


def _numpy_chevallard13(wave_aa: np.ndarray, tau: float) -> np.ndarray:
    """Independent transcription of Chevallard+2013 eqs 9-11 in float64 numpy."""
    n_v = 2.8 / (1.0 + 3.0 * np.sqrt(tau))
    b = 0.3 - 0.05 * tau
    lam_um = np.clip(wave_aa / 1.0e4, 0.1, 2.5)
    n = n_v + b * (lam_um - 0.55)
    return (wave_aa / _V_BAND_ANGSTROM) ** (-n)


@pytest.mark.parametrize("tau", [0.01, 0.1, 0.5, 1.0, 2.0, 4.0])
def test_normalized_at_v_band_for_every_tau(tau):
    """k(5500 A, tau) = 1 for every tau."""
    k = np.asarray(chevallard13(jnp.array([_V_BAND_ANGSTROM]), tau))
    assert_allclose(k, 1.0, atol=1e-12)


@pytest.mark.parametrize("tau", [0.1, 0.5, 2.0])
def test_local_slope_at_v_band_is_minus_n_v(tau):
    """d ln k / d ln lambda at 5500 A equals -n_V(tau), central difference."""
    h = 1e-4
    lam_hi = _V_BAND_ANGSTROM * np.exp(h)
    lam_lo = _V_BAND_ANGSTROM * np.exp(-h)
    k = np.asarray(chevallard13(jnp.array([lam_lo, lam_hi]), tau))
    slope = (np.log(k[1]) - np.log(k[0])) / (2.0 * h)
    n_v = 2.8 / (1.0 + 3.0 * np.sqrt(tau))
    assert abs(slope - (-n_v)) < 1e-4


def test_curve_matches_numpy_transcription():
    """At tau = 1, n(lambda) = 0.7 + 0.25 (lambda/um - 0.55) on 1000-25000 A."""
    wave = np.linspace(1000.0, 25000.0, 1000)
    k = np.asarray(chevallard13(jnp.asarray(wave), 1.0))
    n_expected = 0.7 + 0.25 * (wave / 1.0e4 - 0.55)
    k_expected = (wave / _V_BAND_ANGSTROM) ** (-n_expected)
    assert_allclose(k, k_expected, rtol=1e-12)
    assert_allclose(k, _numpy_chevallard13(wave, 1.0), rtol=1e-12)


def test_slope_flattens_with_tau():
    """The magnitude of the V-band slope decreases monotonically with tau."""
    taus = [0.1, 0.5, 1.0, 2.0, 4.0]
    n_v = [2.8 / (1.0 + 3.0 * np.sqrt(t)) for t in taus]
    assert all(a > b for a, b in itertools.pairwise(n_v))
    # Same ordering from the law itself: steeper (larger k in the UV) at small tau.
    uv = jnp.array([2000.0])
    k_uv = [float(chevallard13(uv, t)[0]) for t in taus]
    assert all(a > b for a, b in itertools.pairwise(k_uv))


def test_edge_index_held_outside_0p1_2p5_um():
    """Outside 0.1-2.5 um the index is the edge value, not an extrapolation."""
    tau = 1.0
    n_v = 0.7
    b = 0.25
    n_blue = n_v + b * (0.1 - 0.55)
    n_red = n_v + b * (2.5 - 0.55)
    k_900 = float(chevallard13(jnp.array([900.0]), tau)[0])
    k_30000 = float(chevallard13(jnp.array([30000.0]), tau)[0])
    assert_allclose(k_900, (900.0 / _V_BAND_ANGSTROM) ** (-n_blue), rtol=1e-12)
    assert_allclose(k_30000, (30000.0 / _V_BAND_ANGSTROM) ** (-n_red), rtol=1e-12)


@pytest.mark.gradient
@pytest.mark.parametrize("tau", [0.0, 0.5])
def test_gradient_wrt_tau_v_finite_at_0_and_0p5(tau):
    """jax.grad with respect to tau_v is finite at tau = 0 and tau = 0.5."""
    wave = jnp.linspace(1000.0, 25000.0, 50)

    def loss(t):
        return jnp.sum(chevallard13(wave, t))

    g = jax.grad(loss)(jnp.asarray(tau))
    assert np.isfinite(float(g))


def test_jit_matches_eager():
    """The jitted law agrees with the eager law."""
    wave = jnp.linspace(1000.0, 25000.0, 100)
    assert_jit_matches_eager(chevallard13, wave, 0.7)


def test_registered_name_and_citation():
    """The registry lists chevallard13 with the Chevallard et al. 2013 citation."""
    assert "chevallard13" in DUST_LAWS
    citation = DUST_LAWS["chevallard13"].citation
    assert "Chevallard" in citation
    assert "2013" in citation
