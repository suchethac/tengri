# SPDX-License-Identifier: BSD-3-Clause
"""Emission-line flux windows are hard, like the spectral-index windows (#2802, #2637).

The spectral indices integrate over exactly ``[lo, hi]`` (#2637). The emission-line
flux windows kept a 1 A sigmoid edge, so a window integral leaked about one
angstrom past each edge. On a piecewise-constant spectrum with its feature edges on
grid nodes, the hard window returns the exact analytic line flux; the sigmoid does
not.

Both tests use a 0.5 A grid. The feature band is ``[5000, 5020]`` with its edges on
nodes, so the piecewise-linear interpolant equals the step exactly inside the band.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.line_measurement import (
    LINE_WINDOW_EDGE_WIDTH,
    LineDef,
    measure_line_flux_jax,
    precompute_line_windows,
)
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

_CONT = 2.0  # continuum level, erg/s/Hz (arbitrary units)
_DELTA = 0.75  # line excess above the continuum on the feature band
_FLO, _FHI = 5000.0, 5020.0


def _step_spectrum() -> tuple[np.ndarray, np.ndarray]:
    wave = np.arange(4000.0, 7000.0 + 0.25, 0.5)
    flux = np.full_like(wave, _CONT)
    flux[(wave >= _FLO) & (wave <= _FHI)] += _DELTA
    return wave, flux


def _line_def() -> LineDef:
    return LineDef(
        name="test_line",
        wavelength=0.5 * (_FLO + _FHI),
        continuum=((4900.0, 4950.0), (5050.0, 5100.0)),
        feature=(_FLO, _FHI),
    )


def test_line_flux_over_step_is_exact_analytic_sum():
    """The line flux of a flat continuum with a step band equals the analytic integral."""
    wave, flux = _step_spectrum()
    line = _line_def()
    lam_c = 0.5 * (_FLO + _FHI)
    width = _FHI - _FLO
    measured = float(measure_line_flux_jax(jnp.asarray(wave), jnp.asarray(flux), line, 0.0))
    exact = _DELTA * width * C_AA / lam_c**2
    np.testing.assert_allclose(measured, exact, rtol=1e-10, atol=0.0)


def test_line_window_integral_over_flat_spectrum_has_exact_width():
    """A flat unit spectrum integrates to the window width, so the normalization is exact."""
    wave = np.arange(4000.0, 7000.0 + 0.25, 0.5)
    ssp_flux = np.ones((1, 1, wave.size))
    line = _line_def()
    precomp = precompute_line_windows(jnp.asarray(wave), jnp.asarray(ssp_flux), (line,))
    norms = np.asarray(precomp.window_norms)
    widths = {
        round(float(hi - lo), 6) for lo, hi in ((4900.0, 4950.0), (5050.0, 5100.0), (_FLO, _FHI))
    }
    assert LINE_WINDOW_EDGE_WIDTH == 0.0, "line windows must be hard (#2637 / #2802)"
    for norm in norms:
        assert any(abs(float(norm) - w) < 1e-9 for w in widths), (float(norm), widths)


def test_line_flux_gradient_wrt_flux_is_exact_and_local():
    """The flux gradient is finite, nonzero in the windows, and matches a finite difference.

    As for the spectral indices (#2637), the window edges are catalog constants, so
    the gradient is taken through the flux only and needs no smoothing.
    """
    import jax

    wave, flux = _step_spectrum()
    line = _line_def()
    wave_j = jnp.asarray(wave)

    def f(fl):
        return measure_line_flux_jax(wave_j, fl, line, 0.0)

    flux_j = jnp.asarray(flux)
    grad = np.asarray(jax.grad(f)(flux_j))
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)
    rng = np.random.default_rng(7)
    direction = rng.standard_normal(flux.size)
    eps = 1e-6
    fd = (float(f(flux_j + eps * direction)) - float(f(flux_j - eps * direction))) / (2 * eps)
    assert float(np.vdot(grad, direction)) == pytest.approx(fd, rel=1e-6)
