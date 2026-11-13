# SPDX-License-Identifier: BSD-3-Clause
"""The qsogen Balmer continuum must be cut at the Balmer edge (#3408).

The old edge was a 0.02 dex logistic, which left 66% of the peak at 3700 A and
20% at 4000 A. The edge is now the analytic step convolved with a Gaussian of
vfwhm = 5000 km/s (upstream qsogen, Temple+2021), so the continuum falls as
erfc with sigma = wavbe * (vfwhm / 2.35) / c ~ 25.9 A.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erfc

from tengri.components.agn.qsogen import _balmer_continuum
from tengri.utils.physics_constants import C_KM_S

pytestmark = pytest.mark.regression_bug

WAVBE = 3646.0
TBC = 15000.0
SIGMA_LAM = WAVBE * (5000.0 / 2.35) / C_KM_S  # ~25.9 A


def _unedged_shape(wave: np.ndarray) -> np.ndarray:
    """Blackbody times (1 - exp(-tau)) without the edge factor (same as the source)."""
    hc_over_k = 1.43877735e8
    x = hc_over_k / (TBC * wave)
    b_nu = wave ** (-3.0) / (np.exp(x) - 1.0)
    tau = (wave / WAVBE) ** 3
    return b_nu * (1.0 - np.exp(-np.clip(tau, 0.0, 50.0)))


def _bc(wave: np.ndarray, bcnorm: float = 1.0) -> np.ndarray:
    cont = np.ones_like(wave)
    out = _balmer_continuum(jnp.asarray(wave), jnp.asarray(cont), bcnorm, TBC, 1.0, WAVBE)
    return np.asarray(out, dtype=float)


def test_sigma_matches_upstream_gaussian_width():
    assert math.isclose(SIGMA_LAM, 25.87, rel_tol=2e-3)


def test_continuum_vanishes_well_above_the_edge():
    wave = np.array([3000.0, 4000.0])
    bc = _bc(wave)
    assert bc[1] / bc[0] < 1e-6


def test_edge_factor_at_3700_is_analytic_erfc():
    wave = np.array([3000.0, 3700.0])
    bc = _bc(wave)
    expected_edge = 0.5 * erfc((3700.0 - WAVBE) / (math.sqrt(2.0) * SIGMA_LAM))
    shape = _unedged_shape(wave)
    expected_ratio = expected_edge * shape[1] / shape[0]
    got_ratio = bc[1] / bc[0]
    assert got_ratio == pytest.approx(expected_ratio, rel=1e-3)


def test_interior_below_3500_has_no_edge_suppression():
    wave = np.array([3000.0, 3500.0])
    bc = _bc(wave)
    shape = _unedged_shape(wave)
    assert (bc[1] / bc[0]) == pytest.approx(shape[1] / shape[0], rel=1e-3)


def test_gradient_wrt_bcnorm_is_finite():
    wave = jnp.asarray(np.array([2500.0, 3000.0, 3600.0, 3700.0, 4200.0]))
    cont = jnp.ones_like(wave)

    def total(b):
        return jnp.sum(_balmer_continuum(wave, cont, b, TBC, 1.0, WAVBE))

    grad = jax.grad(total)(1.0)
    assert np.isfinite(float(grad))
