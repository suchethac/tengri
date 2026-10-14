# SPDX-License-Identifier: BSD-3-Clause
"""The corona's band integral is closed form and agrees with a dense quadrature.

The unabsorbed corona shape ``(E/2)**(1-Gamma) exp(-(E-2)/E_cut)`` integrates over a
band in closed form (an incomplete-gamma difference). The fit surface evaluates it on
every draw, so the closed form replaces a 400-node quadrature. The reference here is a
4000-node Simpson rule, far denser than the 400 nodes it replaced.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import simpson

from tengri.components.xray import xray as X

pytestmark = pytest.mark.contract

#: (Gamma, E_cut [keV]) points: the lopez24 and yang20 defaults (both 1.8, 300 keV),
#: a hard corona past Gamma = 2 (the s <= 0 branch), and a soft, strongly cut-off one.
POINTS = [(1.8, 300.0), (2.2, 300.0), (1.4, 50.0)]
BANDS = [X.CORONA_TOTAL_BAND_KEV, (2.0, 10.0)]


def _quadrature_band_integral(e_lo, e_hi, gamma, e_cut, n=4000):
    """Frequency integral [Hz] of the shape by a 4000-node Simpson rule.

    Simpson rather than the trapezoid: a 4000-node trapezoid is itself only ~1e-6
    accurate at Gamma = 2.2, which would sit on the tolerance. Simpson on the same grid
    agrees with adaptive quadrature to 1e-11.
    """
    energy = np.geomspace(e_lo, e_hi, n)
    shape = (energy / 2.0) ** (1.0 - gamma) * np.exp(-(energy - 2.0) / e_cut)
    return simpson(shape, x=energy * X._KEV_TO_HZ)


@pytest.mark.parametrize("gamma, e_cut", POINTS)
@pytest.mark.parametrize("band", BANDS, ids=["total_0p25_1240", "2_10"])
def test_closed_form_matches_dense_quadrature(gamma, e_cut, band):
    closed = float(X._intrinsic_corona_band_integral(*band, gamma, e_cut))
    dense = _quadrature_band_integral(*band, gamma, e_cut)
    assert closed == pytest.approx(dense, rel=1e-6)


def test_closed_form_is_jit_and_log10_stable_in_float32():
    """A float32 log10 of the closed form matches float64 to float32 round-off."""
    f = jax.jit(lambda g, e: X._intrinsic_corona_band_integral(*X.CORONA_TOTAL_BAND_KEV, g, e))
    with jax.enable_x64(False):
        log32 = float(jnp.log10(f(jnp.float32(1.8), jnp.float32(300.0))))
    log64 = float(np.log10(_quadrature_band_integral(*X.CORONA_TOTAL_BAND_KEV, 1.8, 300.0)))
    assert log32 == pytest.approx(log64, abs=1e-5)


def test_gamma_two_is_continuous_across_the_removable_point():
    """s = 2 - Gamma = 0 is a removable point; both neighbors agree with it."""
    band = X.CORONA_TOTAL_BAND_KEV
    at = float(X._intrinsic_corona_band_integral(*band, 2.0, 300.0))
    left = float(X._intrinsic_corona_band_integral(*band, 2.0 - 1e-4, 300.0))
    right = float(X._intrinsic_corona_band_integral(*band, 2.0 + 1e-4, 300.0))
    assert at == pytest.approx(0.5 * (left + right), rel=1e-6)
