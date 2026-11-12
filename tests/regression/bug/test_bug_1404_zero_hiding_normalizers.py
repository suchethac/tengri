# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1404: a degenerate window fit returned a plausible value, not NaN.

Bug: ``_measure_slope`` divided by ``jnp.maximum(sw, floor)`` and
``jnp.maximum(denom, floor)``. A window holding a single grid point has
``denom == 0``, so the fitted slope was ``0 / floor = 0`` and the index read
as ``-2.0`` (a plausible UV slope). The fix returns NaN where the fit is
undefined.

Scope: only this site. The break, equivalent-width and Dn4000 normalizers are
left on their clamps: a NaN at a zero denominator gives a zero gradient there,
which the #2100 precision tests reject (see the PR body).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectral_indices import SpectralIndexDef, _measure_slope

pytestmark = pytest.mark.regression_bug

UV = SpectralIndexDef(
    name="UV slope",
    index_type="slope",
    continuum=((1250.0, 2600.0),),
    feature=(1250.0, 2600.0),
    units="ang",
)


def _power_law_flux(wave):
    return 1e25 * (wave / 4000.0) ** -1.0


def test_single_point_window_gives_nan_not_minus_two():
    wave = jnp.asarray([1000.0, 1300.0, 3000.0, 5000.0])
    flux = jnp.asarray([1.0, 2.0, 3.0, 4.0]) * 1e25
    assert np.isnan(float(_measure_slope(wave, flux, UV)))


def test_normal_window_bit_identical_to_prefix_formula():
    wave = jnp.asarray(np.linspace(1000.0, 3000.0, 500))
    flux = jnp.asarray(_power_law_flux(np.asarray(wave)))
    got = _measure_slope(wave, flux, UV)
    # Pre-fix formula, verbatim.
    w = ((wave >= 1250.0) & (wave <= 2600.0)).astype(flux.dtype)
    log_wave = jnp.log(jnp.maximum(wave, 1.0))
    log_fnu = jnp.log(jnp.maximum(flux, 1e-50))
    sw = jnp.sum(w)
    sx = jnp.sum(w * log_wave)
    sy = jnp.sum(w * log_fnu)
    sxx = jnp.sum(w * log_wave**2)
    sxy = jnp.sum(w * log_wave * log_fnu)
    floor = jnp.maximum(sw, 1e-30)
    denom = sxx - sx**2 / floor
    expected = (sxy - sx * sy / floor) / jnp.maximum(denom, 1e-30) - 2.0
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    np.testing.assert_allclose(float(got), -3.0, atol=1e-6)


def test_gradient_finite_at_single_point_window():
    wave = jnp.asarray([1000.0, 1300.0, 3000.0, 5000.0])

    def f(flux):
        return jnp.nan_to_num(_measure_slope(wave, flux, UV))

    grad = jax.grad(f)(jnp.asarray([1.0, 2.0, 3.0, 4.0]) * 1e25)
    assert bool(jnp.all(jnp.isfinite(grad)))
