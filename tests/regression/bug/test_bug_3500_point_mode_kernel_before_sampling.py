# SPDX-License-Identifier: BSD-3-Clause
r"""Point mode applies the instrument and velocity kernel before sampling (#3500, #3451).

``project_spectrum`` sampled the model at the pixel centers and then convolved the pixels.
A pixel then depends on its neighbors and on the pixel grid, and model features narrower
than a pixel alias. Measured on the fixture below (R = 2000, sigma_v 0-300 km/s) the old
order was 2e-3 to 8e-3 of peak from the analytic value.

Point mode with a resolution now broadens the model before it is sampled: on the uniform
:math:`\ln\lambda` lattice for a kernel wider than 40 km/s, in closed form for one narrower
than 20 km/s, blended in between. The target of both is

.. math::

    F_\nu(\lambda_c) \propto
    \frac{\mathcal{B}_{\sigma_c}(\lambda L_\nu)(\ln\lambda_c)}{\lambda_c},

the Gaussian in :math:`\ln\lambda` of width :math:`\sigma_c` applied to the model's own
piecewise-linear interpolant :math:`L_\nu` and evaluated at the pixel center
:math:`\lambda_c`.

The reference is ``tests/_lsf_reference.convolve_piecewise_linear``, which integrates the
same interpolant segment by segment and shares no code with the lattice. The fixture is
a continuum with six absorption lines on a rest grid that is dense (0.9 A) across the
optical and sparse outside, as an SSP grid is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.banded import banded_matvec, gaussian_resolution_bands
from tengri.observation.spectrum import (
    apply_lsf,
    broaden_velocity_only,
    compute_spectrum,
    compute_spectrum_conserving,
    lattice_span,
    project_spectrum,
)
from tengri.units import lnu_to_fnu
from tengri.utils.physics_constants import C_KM_S, GAUSSIAN_FWHM_PER_SIGMA
from tests._lsf_reference import convolve_piecewise_linear

pytestmark = pytest.mark.regression_bug

_Z = 0.1
_DL_CM = 1.4e27
_LINES = (
    (3934.8, 0.60, 1.2),
    (3969.6, 0.50, 1.2),
    (4861.3, 0.40, 1.0),
    (5175.0, 0.30, 1.5),
    (5895.0, 0.35, 1.3),
    (6564.6, 0.50, 1.0),
)


def _rest_grid() -> np.ndarray:
    return np.concatenate(
        [
            np.linspace(1000.0, 3500.0, 400, endpoint=False),
            np.arange(3500.0, 7500.0, 0.9),
            np.linspace(7500.0, 20000.0, 600),
        ]
    )


def _sed(wave: np.ndarray) -> np.ndarray:
    absorbed = sum(d * np.exp(-0.5 * ((wave - c) / w) ** 2) for c, d, w in _LINES)
    return 1.0e28 * (wave / 5500.0) ** -0.7 * (1.0 - absorbed)


_WAVE_REST = _rest_grid()
_SED = _sed(_WAVE_REST)
_WAVE_OBS = np.linspace(4000.0, 7900.0, 1800)


def _width_ln(resolution, sigma_v_kms: float) -> np.ndarray:
    sigma_inst = C_KM_S / (GAUSSIAN_FWHM_PER_SIGMA * np.asarray(resolution, dtype=float))
    width = np.sqrt(sigma_inst**2 + sigma_v_kms**2) / C_KM_S
    return np.broadcast_to(width, _WAVE_OBS.shape)


def _reference(resolution, sigma_v_kms: float, wave_rest=_WAVE_REST, sed=_SED) -> np.ndarray:
    dimming = float(lnu_to_fnu(jnp.ones(()), _DL_CM, _Z))
    convolved = convolve_piecewise_linear(
        wave_rest, sed, _WAVE_OBS, _Z, _width_ln(resolution, sigma_v_kms), quantity="nufnu"
    )
    return dimming * np.asarray(convolved)


def _project(resolution, sigma_v_kms, wave_rest=_WAVE_REST, sed=_SED, wave_obs=_WAVE_OBS, **kw):
    return project_spectrum(
        jnp.asarray(sed),
        jnp.asarray(wave_rest),
        jnp.asarray(wave_obs),
        _Z,
        _DL_CM,
        resolution=resolution,
        sigma_v_kms=sigma_v_kms,
        **kw,
    )


def _max_rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


@pytest.mark.parametrize(
    ("resolution", "sigma_v_kms"),
    [(2000.0, 0.0), (2000.0, 50.0), (2000.0, 200.0), (2000.0, 300.0), (5000.0, 200.0)],
)
def test_point_mode_matches_the_analytic_value(resolution, sigma_v_kms):
    reference = _reference(resolution, sigma_v_kms)
    # Measured: 1.9e-5, 9.9e-6, 9.0e-7, 4.1e-7 and 9.8e-7 of peak, the lattice's accuracy.
    assert _max_rel(_project(resolution, sigma_v_kms), reference) < 5e-5


@pytest.mark.parametrize(("resolution", "sigma_v_kms"), [(2000.0, 0.0), (2000.0, 300.0)])
def test_sampling_then_convolving_is_measurably_different(resolution, sigma_v_kms):
    """The fixture can see the defect: the old order is off by more than 1e-3 of peak."""
    sampled = compute_spectrum(
        jnp.asarray(_SED), jnp.asarray(_WAVE_REST), jnp.asarray(_WAVE_OBS), _Z, _DL_CM
    )
    old = apply_lsf(sampled, jnp.asarray(_WAVE_OBS), resolution, sigma_v_kms=sigma_v_kms)
    # Measured: 7.5e-3 at sigma_v = 0 and 2.3e-3 at 300 km/s.
    assert _max_rel(old, _reference(resolution, sigma_v_kms)) > 1e-3


def test_result_does_not_depend_on_how_the_model_grid_is_refined():
    """Extra nodes on the same interpolant change nothing (#3451)."""
    midpoints = 0.5 * (_WAVE_REST[1:] + _WAVE_REST[:-1])
    refined = np.sort(np.concatenate([_WAVE_REST, midpoints]))
    refined_sed = np.interp(refined, _WAVE_REST, _SED)
    base = _project(2000.0, 200.0)
    # Measured: 1.5e-13 of peak.
    assert _max_rel(_project(2000.0, 200.0, wave_rest=refined, sed=refined_sed), base) < 1e-10


def test_a_pixel_does_not_depend_on_the_other_pixels():
    every_third = _WAVE_OBS[::3]
    full = np.asarray(_project(2000.0, 200.0))
    subset = np.asarray(_project(2000.0, 200.0, wave_obs=every_third))
    # Measured: 3.1e-13 of peak; the two calls build lattices that end on different cells.
    np.testing.assert_allclose(subset, full[::3], rtol=1e-10)


def test_pad_and_segment_count_do_not_enter_for_a_constant_resolution():
    base = np.asarray(_project(2000.0, 200.0))
    other = np.asarray(_project(2000.0, 200.0, n_bins=3, lsf_pad_pixels=8))
    np.testing.assert_array_equal(other, base)


@pytest.mark.parametrize("resolution", [100.0, 30.0])
def test_a_low_resolution_kernel_matches_the_analytic_value(resolution):
    """sigma_inst is 1273 km/s at R = 100 and 4244 km/s at R = 30."""
    # Measured: 2.3e-8 and 2.2e-5 of peak.
    assert _max_rel(_project(resolution, 0.0), _reference(resolution, 0.0)) < 1e-4


@pytest.mark.parametrize(("resolution", "sigma_v_kms"), [(10000.0, 0.0), (5000.0, 0.0)])
def test_a_kernel_narrower_than_the_lattice_is_read_in_closed_form(resolution, sigma_v_kms):
    """12.7 km/s is below the lattice spacing of 15 km/s; the lattice alone is 3.5e-4 off."""
    reference = _reference(resolution, sigma_v_kms)
    # Measured: 2.1e-13 at 12.7 km/s (closed form) and 2.0e-5 at 25.5 km/s (blended).
    bound = 1e-10 if resolution == 10000.0 else 1e-4
    assert _max_rel(_project(resolution, sigma_v_kms), reference) < bound


@pytest.mark.parametrize("sigma_v_kms", [5.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 150.0])
def test_unresolved_instrument_follows_sigma_v_at_every_width(sigma_v_kms):
    """Library coarser than the instrument: the kernel is sigma_v alone, through the blend."""
    reference = _reference(1.0e9, sigma_v_kms)
    out = _project(2000.0, sigma_v_kms, sigma_lib_kms=500.0)
    # Measured: 1.6e-12 and 7.0e-13 below 20 km/s, at most 5.9e-5 from 25 to 150 km/s.
    assert _max_rel(out, reference) < 2e-4


def test_per_pixel_resolution_matches_the_analytic_value():
    resolution = np.linspace(1500.0, 4000.0, _WAVE_OBS.size)
    reference = _reference(resolution, 100.0)
    # Measured: 5.8e-4 of peak with the default 16 segments, 1.6e-4 with 64. The segments
    # share one width each, which is the approximation.
    assert _max_rel(_project(jnp.asarray(resolution), 100.0), reference) < 2e-3
    assert _max_rel(_project(jnp.asarray(resolution), 100.0, n_bins=64), reference) < 5e-4


def test_zero_width_pixels_take_the_point_sample():
    """Library coarser than the instrument and sigma_v = 0: no kernel, the interpolant itself."""
    sampled = compute_spectrum(
        jnp.asarray(_SED), jnp.asarray(_WAVE_REST), jnp.asarray(_WAVE_OBS), _Z, _DL_CM
    )
    out = _project(2000.0, 0.0, sigma_lib_kms=500.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(sampled), rtol=1e-12)


def test_sigma_v_gradient_matches_a_finite_difference():
    def loss(sigma_v):
        return jnp.sum(_project(2000.0, sigma_v) ** 2) * 1.0e54

    sigma_v, step = 150.0, 1.0e-1
    grad = float(jax.grad(loss)(sigma_v))
    reference = float((loss(sigma_v + step) - loss(sigma_v - step)) / (2.0 * step))
    assert np.isfinite(grad), "a positive kernel width has a finite derivative"
    assert grad != 0.0, "150 km/s broadens every absorption line of the fixture"
    # Measured: 1.7e-7 relative.
    np.testing.assert_allclose(grad, reference, rtol=1e-5)


def test_traced_sigma_v_agrees_with_the_concrete_call():
    """Under jit the lattice is passed in, as the model does; the value is the same."""
    span = lattice_span(_WAVE_OBS, _Z, _Z)
    traced = jax.jit(lambda sigma_v: _project(2000.0, sigma_v, lattice_window=span))(200.0)
    assert _max_rel(traced, _project(2000.0, 200.0)) < 1e-10


def test_conserving_matrix_branch_is_the_same_composition_as_before():
    """The resolution-matrix branch is untouched: resample, sigma_v on the pixels, banded LSF."""
    wave_obs = jnp.asarray(_WAVE_OBS)
    matrix = gaussian_resolution_bands(wave_obs, np.linspace(2000.0, 5000.0, _WAVE_OBS.size))
    sed, wave_rest = jnp.asarray(_SED), jnp.asarray(_WAVE_REST)
    for conserving in (True, False):
        resample = compute_spectrum_conserving if conserving else compute_spectrum
        flux = resample(sed, wave_rest, wave_obs, _Z, _DL_CM)
        flux = broaden_velocity_only(flux, wave_obs, 80.0, 16, pad_pixels=None)
        manual = banded_matvec(matrix.offsets, matrix.data, flux)
        out = project_spectrum(
            sed,
            wave_rest,
            wave_obs,
            _Z,
            _DL_CM,
            resolution_matrix=matrix,
            sigma_v_kms=80.0,
            conserving=conserving,
        )
        np.testing.assert_array_equal(np.asarray(out), np.asarray(manual))
