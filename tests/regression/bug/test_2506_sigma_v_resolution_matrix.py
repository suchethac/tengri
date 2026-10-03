# SPDX-License-Identifier: BSD-3-Clause
"""#2506: sigma_v_kms must broaden the model on the resolution-matrix path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401  (float64)
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.spectrum import project_spectrum

pytestmark = pytest.mark.regression_bug

C_KMS = 299792.458


def _setup(n=4000, z=0.03):
    wave = jnp.arange(3600.0, 3600.0 + 0.8 * n, 0.8)  # DESI-like LINEAR grid
    wave_rest = jnp.geomspace(3000.0, 10500.0, 200000)  # ~0.02-0.06 A sampling
    # Narrow (0.1 A rest) absorption lines every 150 A obs so broadening is measurable.
    centers = np.arange(3600.0, 6700.0, 150.0) / (1 + z)
    sed = 1.0 - sum(0.5 * np.exp(-0.5 * ((np.asarray(wave_rest) - c) / 0.1) ** 2) for c in centers)
    return wave, wave_rest, jnp.asarray(sed), z


def _project(sed, wave_rest, wave, z, sigma_v, **kw):
    return project_spectrum(
        sed, wave_rest, wave, z, 1e26, conserving=True, sigma_v_kms=sigma_v, **kw
    )


def test_sigma_v_changes_matrix_path_spectrum_and_gradient():
    wave, wave_rest, sed, z = _setup()
    bm = gaussian_resolution_bands(wave, 3500.0, n_diag=11)
    a = _project(sed, wave_rest, wave, z, 80.0, resolution_matrix=bm)
    b = _project(sed, wave_rest, wave, z, 150.0, resolution_matrix=bm)
    assert float(jnp.max(jnp.abs(b - a)) / jnp.max(jnp.abs(a))) > 1e-2
    g = jax.grad(
        lambda s: jnp.sum(_project(sed, wave_rest, wave, z, s, resolution_matrix=bm) ** 2)
    )(100.0)
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_sigma_v_zero_is_identity_on_matrix_path():
    """Existing DESI fits (sigma_v unset/0) must not move."""
    from tengri.observation.banded import banded_matvec
    from tengri.observation.spectrum import compute_spectrum_conserving

    wave, wave_rest, sed, z = _setup()
    bm = gaussian_resolution_bands(wave, 3500.0, n_diag=11)
    new = _project(sed, wave_rest, wave, z, 0.0, resolution_matrix=bm)
    ref = banded_matvec(
        bm.offsets, bm.data, compute_spectrum_conserving(sed, wave_rest, wave, z, 1e26)
    )
    np.testing.assert_allclose(np.asarray(new), np.asarray(ref), rtol=1e-12, atol=0)


@pytest.mark.parametrize("sigma_v", [20.0, 60.0, 150.0])
def test_matrix_path_sigma_v_matches_gaussian_path_linear_grid(sigma_v):
    """A Gaussian-equivalent matrix at R acts on the pixel-binned model, like apply_lsf on pixels.

    The resolution matrix maps the model binned onto the pixels to the observed
    pixels (Bolton & Schlegel 2010), so the matrix path bins first and
    broadens second. It therefore equals ``apply_lsf`` run on the pixel-binned
    model (same R and sigma_v) to the banded-kernel discretization, 1e-2.
    The Gaussian-LSF path of ``project_spectrum`` under the pixel mean
    broadens the model grid first (#2530); the two orders differ by the
    pixel-binning error of the 0.1 Angstrom lines in 0.8 Angstrom pixels of
    this fixture (measured 1.1e-2, 2.7e-3, 4.8e-4 for sigma_v = 20, 60, 150
    km/s), bounded at 2e-2.
    """
    from tengri.observation.spectrum import apply_lsf, compute_spectrum_conserving

    wave, wave_rest, sed, z = _setup()
    bm = gaussian_resolution_bands(wave, 3500.0, n_diag=31)
    m = np.asarray(_project(sed, wave_rest, wave, z, sigma_v, resolution_matrix=bm))
    binned = compute_spectrum_conserving(sed, wave_rest, wave, z, 1e26)
    pixel_order = np.asarray(
        apply_lsf(binned, wave, 3500.0, sigma_lib_kms=0.0, sigma_v_kms=sigma_v)
    )
    model_order = np.asarray(
        _project(sed, wave_rest, wave, z, sigma_v, resolution=3500.0, sigma_lib_kms=0.0)
    )
    core = slice(200, -200)  # avoid FFT/band edges

    def rel(a, b):
        return np.max(np.abs(a[core] - b[core])) / np.max(np.abs(b[core]))

    assert rel(m, pixel_order) < 1e-2, rel(m, pixel_order)
    assert rel(m, model_order) < 2e-2, rel(m, model_order)


def test_measured_line_width_blue_and_red(sigma_v=100.0):
    """Width of an isolated line = sqrt(sigma_R^2 + sigma_v^2) at both ends of a linear
    grid (#1742 class)."""
    wave, wave_rest, sed, z = _setup()
    bm = gaussian_resolution_bands(wave, 3500.0, n_diag=41)
    flux = np.asarray(_project(sed, wave_rest, wave, z, sigma_v, resolution_matrix=bm))
    out = 1.0 - flux / np.median(flux)  # continuum-normalize (project_spectrum returns f_nu)
    w = np.asarray(wave)
    sig_R = C_KMS / (2.3548 * 3500.0)
    for lam0 in (3750.0, 6450.0):
        # Expected width: instrument + sigma_v + intrinsic line (0.1 A rest) + 0.8 A pixel boxcar.
        sig_line = 0.1 * (1 + z) / lam0 * C_KMS
        sig_box = (0.8 / np.sqrt(12.0)) / lam0 * C_KMS
        expect = np.sqrt(sig_R**2 + sigma_v**2 + sig_line**2 + sig_box**2)
        sel = np.abs(w - lam0) < 8.0
        prof = out[sel] - np.median(out[np.abs(w - lam0) < 40.0])
        prof = np.clip(prof, 0, None)
        mu = np.sum(w[sel] * prof) / np.sum(prof)
        var = np.sum((w[sel] - mu) ** 2 * prof) / np.sum(prof)
        sigma_kms = np.sqrt(var) / mu * C_KMS
        assert abs(sigma_kms / expect - 1) < 0.05, (lam0, sigma_kms, expect)
