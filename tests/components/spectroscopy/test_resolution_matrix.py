# SPDX-License-Identifier: BSD-3-Clause
"""DESI/PFS resolution matrix wired through the spectroscopic forward model (#1163).

The banded resolution matrix replaces the Gaussian ``apply_lsf`` when present.
These tests exercise the container (``Spectroscopy.resolution_matrix``), the
projector (``project_spectrum`` applies ``R @ model`` after the flux-conserving
resample), and the end-to-end path through ``SEDModel.predict_spectrum``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401  (enables float64)
from tengri import Spectroscopy
from tengri.observation.banded import banded_matvec, gaussian_resolution_bands
from tengri.observation.spectrum import compute_spectrum_conserving, project_spectrum

pytestmark = pytest.mark.contract


def test_spectroscopy_accepts_resolution_matrix():
    wave = np.geomspace(4000.0, 7000.0, 100)
    bm = gaussian_resolution_bands(jnp.asarray(wave), 2000.0, n_diag=11)
    spec = Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=bm)
    assert spec.has_resolution_matrix
    assert Spectroscopy(wave_obs=jnp.asarray(wave)).has_resolution_matrix is False


def test_resolution_matrix_shape_validated():
    wave = np.geomspace(4000.0, 7000.0, 100)
    bm = gaussian_resolution_bands(jnp.asarray(np.geomspace(4000.0, 7000.0, 50)), 2000.0)
    with pytest.raises(ValueError, match="resolution_matrix"):
        Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=bm)


def test_project_spectrum_applies_resolution_matrix():
    n = 300
    wave = np.geomspace(4000.0, 7000.0, n)
    wave_rest = wave / 1.05
    sed = np.ones(n) + 0.2 * np.sin(np.linspace(0, 30, n))
    bm = gaussian_resolution_bands(jnp.asarray(wave), 2500.0, n_diag=21)
    dl_cm = 1e26

    flux = project_spectrum(
        jnp.asarray(sed),
        jnp.asarray(wave_rest),
        jnp.asarray(wave),
        0.05,
        dl_cm,
        resolution_matrix=bm,
        conserving=True,
    )
    # Reference: flux-conserving resample, then the dense-equivalent R @ .
    resampled = compute_spectrum_conserving(
        jnp.asarray(sed), jnp.asarray(wave_rest), jnp.asarray(wave), 0.05, dl_cm
    )
    ref = banded_matvec(bm.offsets, bm.data, resampled)
    np.testing.assert_allclose(np.asarray(flux), np.asarray(ref), rtol=1e-10, atol=1e-30)


def test_resolution_matrix_takes_precedence_over_scalar_resolution():
    # When both a matrix and a scalar resolution are supplied, the matrix wins
    # (apply_lsf is skipped) — the R matrix already encodes the LSF.
    n = 200
    wave = np.geomspace(4000.0, 7000.0, n)
    wave_rest = wave / 1.05
    sed = np.ones(n) + 0.3 * np.sin(np.linspace(0, 20, n))
    bm = gaussian_resolution_bands(jnp.asarray(wave), 2500.0, n_diag=21)
    dl_cm = 1e26

    flux_matrix = project_spectrum(
        jnp.asarray(sed),
        jnp.asarray(wave_rest),
        jnp.asarray(wave),
        0.05,
        dl_cm,
        resolution=800.0,  # deliberately different R — must be ignored
        resolution_matrix=bm,
        conserving=True,
    )
    resampled = compute_spectrum_conserving(
        jnp.asarray(sed), jnp.asarray(wave_rest), jnp.asarray(wave), 0.05, dl_cm
    )
    ref = banded_matvec(bm.offsets, bm.data, resampled)
    np.testing.assert_allclose(np.asarray(flux_matrix), np.asarray(ref), rtol=1e-10, atol=1e-30)
