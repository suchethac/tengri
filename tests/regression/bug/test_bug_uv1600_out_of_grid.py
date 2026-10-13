# SPDX-License-Identifier: BSD-3-Clause
"""Regression test: the 1600 A UV luminosity when the grid does not reach 1600 A.

Bug: ``jnp.interp(1600.0, wave, sed)`` clamps to the edge value when ``wave``
does not cover 1600 A. A grid starting at 1700 A therefore reported ``sed[0]``
as the UV luminosity, and IRX built on it was a plausible wrong number with no
signal. Found under the silent-failure audit that followed b825859e2 (#2937).

Behavior (decided from the concrete grid, one branch per case):

- Grid spans 1600 A: linear interpolation, unchanged and bit-identical.
- Grid stops short of 1600 A with at least two Calzetti 1994 UV-window points:
  power-law extrapolation of ``L_nu`` in log-log space, with a UserWarning.
- Fewer than two window points: the nearest edge value (the legacy clamp), with
  a UserWarning. This is the only case with no better estimate.

Mutation requirements:
1. Power-law grid: the extrapolated L_1600 matches the analytic value (dies if
   the fit, the weights, or the anchor is wrong).
2. Extrapolation and edge-fallback paths both warn (dies if the warning is dropped).
3. In-grid value is bit-identical to ``jnp.interp`` and emits no warning.
4. The gradient w.r.t. the SED is finite AND non-zero, in-grid and extrapolated.
5. No UV points: the edge value, not an extrapolated one.
6. float32: the log form stays finite under ``jax.enable_x64(False)``.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import log10_magnitude
from tengri.utils.sed_quantities import (
    LOG10_NU_1600,
    compute_irx,
    compute_log_uv_luminosity_1600,
    compute_uv_luminosity_1600,
)

pytestmark = pytest.mark.regression_bug

L_TIR_LSUN = 1.0e10
BETA = -1.5  # L_lambda ~ lambda^BETA, so L_nu ~ lambda^(BETA + 2)
A_NORM = 1.0e27  # L_nu at 1e4 A [erg/s/Hz]


def _power_law_sed(wave: np.ndarray) -> np.ndarray:
    """A pure power law in L_nu with slope BETA + 2 (no curvature, no floor)."""
    return A_NORM * (wave / 1.0e4) ** (BETA + 2.0)


def _analytic_l_uv_1600() -> float:
    """Exact nu*L_nu at 1600 A for the power law above [erg/s]."""
    return (C_AA / 1600.0) * A_NORM * (1600.0 / 1.0e4) ** (BETA + 2.0)


def test_power_law_extrapolation_matches_analytic_value():
    wave = np.geomspace(1700.0, 10000.0, 400)
    sed = jnp.asarray(_power_law_sed(wave))
    with pytest.warns(UserWarning, match="extrapolated along a power law"):
        l_uv = float(compute_uv_luminosity_1600(sed, jnp.asarray(wave)))
    np.testing.assert_allclose(l_uv, _analytic_l_uv_1600(), rtol=1e-6)


def test_power_law_extrapolation_log_form_and_irx_are_finite_and_consistent():
    wave = np.geomspace(1700.0, 10000.0, 400)
    sed = jnp.asarray(_power_law_sed(wave))
    with pytest.warns(UserWarning, match="extrapolated along a power law"):
        log_uv = float(compute_log_uv_luminosity_1600(sed, jnp.asarray(wave)))
    np.testing.assert_allclose(10.0**log_uv, _analytic_l_uv_1600(), rtol=1e-6)
    irx = float(compute_irx(jnp.asarray(L_TIR_LSUN), log_l_uv_erg=jnp.asarray(log_uv)))
    assert np.isfinite(irx)


def test_in_grid_value_is_bit_identical_to_interp_and_silent():
    wave = np.geomspace(91.0, 1.0e6, 2000)
    sed = jnp.asarray(1.0e27 * (wave / 1.0e4) ** -1.5 + 1.0e25)
    wave_j = jnp.asarray(wave)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        got = compute_uv_luminosity_1600(sed, wave_j)
        got_log = compute_log_uv_luminosity_1600(sed, wave_j)
    assert not [w for w in record if issubclass(w.category, UserWarning)]
    expected = (C_AA / 1600.0) * jnp.interp(1600.0, wave_j, sed)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    expected_log = log10_magnitude(jnp.interp(1600.0, wave_j, sed)) + LOG10_NU_1600
    np.testing.assert_array_equal(np.asarray(got_log), np.asarray(expected_log))


@pytest.mark.parametrize("grid", ["in_grid", "extrapolated"])
def test_gradient_wrt_sed_is_finite_and_nonzero(grid):
    if grid == "in_grid":
        wave = np.geomspace(91.0, 1.0e6, 2000)
        sed_np = 1.0e27 * (wave / 1.0e4) ** -1.5 + 1.0e25
    else:
        wave = np.geomspace(1700.0, 10000.0, 400)
        sed_np = _power_law_sed(wave)
    wave_j = jnp.asarray(wave)
    sed = jnp.asarray(sed_np)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grad_lin = jax.grad(lambda s: compute_uv_luminosity_1600(s, wave_j))(sed)
        grad_log = jax.grad(lambda s: compute_log_uv_luminosity_1600(s, wave_j))(sed)
    for grad in (grad_lin, grad_log):
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.sum(jnp.abs(grad))) > 0.0


def test_no_uv_points_falls_back_to_edge_value_with_warning():
    wave = np.geomspace(2700.0, 10000.0, 300)  # nothing in 1268-2580 A
    sed = jnp.asarray(_power_law_sed(wave))
    with pytest.warns(UserWarning, match="nearest edge value"):
        l_uv = float(compute_uv_luminosity_1600(sed, jnp.asarray(wave)))
    np.testing.assert_allclose(l_uv, (C_AA / 1600.0) * float(sed[0]), rtol=1e-12)


def test_float32_log_extrapolation_is_finite():
    wave_np = np.geomspace(1700.0, 10000.0, 400)
    sed_np = _power_law_sed(wave_np)
    with jax.enable_x64(False):
        wave = jnp.asarray(wave_np, dtype=jnp.float32)
        sed = jnp.asarray(sed_np, dtype=jnp.float32)
        with pytest.warns(UserWarning, match="extrapolated along a power law"):
            log_uv = compute_log_uv_luminosity_1600(sed, wave)
        assert log_uv.dtype == jnp.float32
        assert bool(jnp.isfinite(log_uv))
        np.testing.assert_allclose(float(log_uv), np.log10(_analytic_l_uv_1600()), rtol=1e-4)
