# SPDX-License-Identifier: BSD-3-Clause
"""#1742: velocity_broaden took its pixel scale from the bluest pixel pair.

A velocity kernel has constant width in ``ln lambda``, and the pixel scale was
read as ``log(wave[1] / wave[0])`` — one pair, at the blue edge. That is exact
on a log-uniform grid and wrong on a linear one, where
``d(ln lambda) ~ 1 / lambda``: the kernel is then set by the bluest pixel while
the feature being broadened sits elsewhere, and the recovered width comes out
low by exactly ``wave[0] / lambda_feature``.

The docstring made it likely rather than unlikely. Its Parameters block — the
one describing the argument a caller is about to pass — said "Must be uniformly
spaced", while Notes said "uniform log-wavelength spacing". Following the
former gave a silently wrong answer: 10% low over 4500-5500 A, and a factor of
three over a 3000-10000 A optical spectrum. The error scales with the
wavelength *range*, not the pixel count, so refining the grid did not help, and
a fitted stellar velocity dispersion inherited it smoothly with nothing to
indicate a problem.

The scale is now taken across the whole grid: unchanged on the documented
log-uniform grid, and centered rather than blue-edge-pinned on a linear one.
A residual lambda dependence remains on a linear grid and is *inherent* — the
local ``d(ln lambda)`` varies by the fractional bandwidth, so one kernel width
cannot serve every wavelength. That is what the log-uniform contract exists
for; the fix removes the systematic bias, not the spread.

``velocity_broaden`` is ``jax.jit``-compiled and the spacing is a traced value,
so it cannot raise on a bad grid — degrading gracefully is the available remedy.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectrum import velocity_broaden

pytestmark = pytest.mark.regression_bug

_C_KM_S = 299792.458
_LAMBDA_0 = 5000.0
_N_PIX = 4096


def _recovered_sigma_kms(wave: jnp.ndarray, sigma_km_s: float) -> float:
    """Broaden a near-delta line at 5000 A and measure its width by second moment."""
    log_wave = np.log(np.asarray(wave))
    idx = int(np.argmin(np.abs(np.asarray(wave) - _LAMBDA_0)))
    line_width = log_wave[idx + 1] - log_wave[idx]

    flux_in = jnp.exp(-0.5 * ((jnp.log(wave) - float(np.log(_LAMBDA_0))) / line_width) ** 2)
    flux_in = flux_in / jnp.sum(flux_in)

    flux_out = np.asarray(velocity_broaden(flux_in, wave, sigma_km_s))
    weights = flux_out / flux_out.sum()
    mean = float((log_wave * weights).sum())
    return float(np.sqrt(((log_wave - mean) ** 2 * weights).sum())) * _C_KM_S


def _log_uniform_grid() -> jnp.ndarray:
    return jnp.exp(jnp.linspace(float(np.log(4500.0)), float(np.log(5500.0)), _N_PIX))


def _linear_grid() -> jnp.ndarray:
    return jnp.linspace(4500.0, 5500.0, _N_PIX)


@pytest.mark.parametrize("sigma_km_s", [100.0, 200.0, 300.0, 500.0])
def test_log_uniform_grid_is_exact(sigma_km_s):
    """The documented grid recovers the requested width.

    The input line is one pixel wide, so its own width adds in quadrature.
    """
    wave = _log_uniform_grid()
    dlnwave = float(jnp.log(wave[1] / wave[0]))
    expected = float(np.hypot(sigma_km_s, dlnwave * _C_KM_S))

    np.testing.assert_allclose(_recovered_sigma_kms(wave, sigma_km_s), expected, rtol=1e-4)


@pytest.mark.parametrize("sigma_km_s", [100.0, 300.0])
def test_linear_grid_is_not_biased_by_the_blue_edge(sigma_km_s):
    """A linear grid must not under-broaden by ``wave[0] / lambda_feature``.

    That factor is 4500/5000 = 0.900 for this grid, and it is what the old
    single-pair scale produced: 90.2 km/s for a requested 100.
    """
    recovered = _recovered_sigma_kms(_linear_grid(), sigma_km_s)

    np.testing.assert_allclose(recovered, sigma_km_s, rtol=0.02)
    assert recovered > 0.95 * sigma_km_s, (
        f"recovered {recovered:.2f} km/s for a requested {sigma_km_s:.0f}; a ratio near "
        f"{4500.0 / _LAMBDA_0:.3f} means the scale is being read off the blue edge again"
    )


def test_linear_grid_is_unbiased_on_average_across_the_band():
    """The mean scale centers the error; it cannot remove the lambda dependence.

    On a linear grid the *local* ``d(ln lambda)`` varies as ``1 / lambda`` — by
    +/-10% across 4500-5500 A — so a single kernel width cannot be right at
    every wavelength. That residual spread is inherent to convolving a velocity
    kernel on a linear grid, and is exactly why the log-uniform grid is the
    documented contract.

    What the fix removes is the *systematic* part: the old scale was pinned to
    the blue edge, so every line came back low, by more the redder it sat. Now
    the error straddles zero across the band. This test pins that centering, and
    would fail again if the scale returned to any single pixel pair.
    """
    wave = _linear_grid()
    log_wave = np.log(np.asarray(wave))
    requested = 300.0

    widths = []
    for lambda_line in (4700.0, 5000.0, 5300.0):
        idx = int(np.argmin(np.abs(np.asarray(wave) - lambda_line)))
        line_width = log_wave[idx + 1] - log_wave[idx]
        flux_in = jnp.exp(-0.5 * ((jnp.log(wave) - float(np.log(lambda_line))) / line_width) ** 2)
        flux_in = flux_in / jnp.sum(flux_in)
        flux_out = np.asarray(velocity_broaden(flux_in, wave, requested))
        weights = flux_out / flux_out.sum()
        mean = float((log_wave * weights).sum())
        widths.append(float(np.sqrt(((log_wave - mean) ** 2 * weights).sum())) * _C_KM_S)

    np.testing.assert_allclose(float(np.mean(widths)), requested, rtol=0.02)
    assert min(widths) < requested < max(widths), (
        "the error should straddle the requested width rather than sit entirely "
        f"below it (blue-edge bias): {widths} for a requested {requested:.0f} km/s"
    )
