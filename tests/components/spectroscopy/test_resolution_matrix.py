# SPDX-License-Identifier: BSD-3-Clause
"""DESI/PFS resolution matrix wired through the spectroscopic forward model (#1163).

The banded resolution matrix replaces the Gaussian ``apply_lsf`` when present.
These tests exercise the container (``Spectroscopy.resolution_matrix``), the
projector (``project_spectrum`` applies ``R @ model`` after the flux-conserving
resample), and the end-to-end path through ``SEDModel.predict_spectrum``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401  (enables float64)
from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy
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


def _spec_model(ssp, wave, *, matrix):
    kw = {"wave_obs": jnp.asarray(wave), "resample": "conserving"}
    if matrix is not None:
        kw["resolution_matrix"] = matrix
    obs = Observation(spectroscopy=Spectroscopy(**kw))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )


def test_predict_spectrum_applies_resolution_matrix_end_to_end(synthetic_ssp):
    # Rest-frame coverage of synthetic_ssp is 3000-10000 A; at z=0.05 the pixel
    # grid 4000-7000 A maps to rest 3810-6667 A, comfortably in range.
    wave = np.geomspace(4000.0, 7000.0, 160)
    # R=300 gives sigma ~ 0.4 pixels on this grid — a genuine blur (~4% change),
    # unlike R=2500 which is sub-pixel (sigma ~ 0.05 pix) and numerically the
    # identity, making a "non-no-op" check meaningless. See the #1163 probe.
    bm = gaussian_resolution_bands(jnp.asarray(wave), 300.0, n_diag=21)

    model_matrix = _spec_model(synthetic_ssp, wave, matrix=bm)
    model_plain = _spec_model(synthetic_ssp, wave, matrix=None)  # conserving, no LSF
    p = dict(model_matrix.spec.sample(jax.random.PRNGKey(0)))

    flux_full = np.asarray(model_matrix.predict_spectrum(p))
    flux_plain = np.asarray(model_plain.predict_spectrum(p))

    assert np.all(np.isfinite(flux_full))
    assert flux_full.shape == wave.shape
    ref = np.asarray(banded_matvec(bm.offsets, bm.data, jnp.asarray(flux_plain)))
    d_full_plain = float(np.max(np.abs(flux_full - flux_plain) / np.abs(flux_plain)))
    d_full_ref = float(np.max(np.abs(flux_full - ref) / np.abs(ref)))
    d_ref_plain = float(np.max(np.abs(ref - flux_plain) / np.abs(flux_plain)))
    # Non-no-op: the matrix must visibly smooth the spectrum (guards a silent
    # drop where the matrix model reuses the plain kernel from the cache).
    # atol=0.0 is mandatory: the flux is physical F_nu at a cosmological
    # distance (~1e-15), so the default atol=1e-8 would swamp the real
    # relative difference and read every spectrum as "close".
    assert not np.allclose(flux_full, flux_plain, rtol=1e-3, atol=0.0), (
        f"resolution_matrix had no effect end-to-end — silently dropped? "
        f"d(full,plain)={d_full_plain:.3e} d(full,R@plain)={d_full_ref:.3e} "
        f"d(R@plain,plain)={d_ref_plain:.3e}"
    )
    # Exactness: predict_spectrum applies R @ (conserving resample) — compare to
    # the plain (no-LSF) spectrum pushed through the same banded operator.
    np.testing.assert_allclose(flux_full, ref, rtol=1e-9, atol=1e-30)
