# SPDX-License-Identifier: BSD-3-Clause
"""Library-LSF deconvolution of a banded resolution matrix.

SSP spectra already carry the stellar library's LSF (sigma_lib); applying the
DESI resolution matrix R on top double-counts it. ``deconvolve_library_lsf``
removes a Gaussian sigma_lib from R via a second-order expansion, and
``row_sigma_kms`` measures the Gaussian-equivalent velocity width of each row
of a banded LSF so the deconvolution (and tests of it) can be checked in
physical units.
"""

import warnings

import numpy as np
import pytest

import tengri  # noqa: F401
from tengri.observation.banded import (
    BandedMatrix,
    deconvolve_library_lsf,
    gaussian_resolution_bands,
    row_sigma_kms,
)

C = 299792.458
WAVE = np.arange(5000.0, 5000.0 + 0.8 * 2000, 0.8)


def _gaussian_bm_with_zeroed_edges(n_edge=5):
    """A Gaussian resolution matrix with its first/last ``n_edge`` columns
    zeroed out, standing in for the masked/edge pixels a real DESI
    resolution matrix carries."""
    bm = gaussian_resolution_bands(WAVE, 3000.0, n_diag=41)
    data = np.array(bm.data)
    data[:, :n_edge] = 0.0
    data[:, -n_edge:] = 0.0
    return BandedMatrix(offsets=bm.offsets, data=data), bm, n_edge


@pytest.mark.limit
def test_row_sigma_matches_gaussian_input():
    """row_sigma_kms recovers R's input sigma_v = c/(2.3548*R) exactly.

    Limit: a pure Gaussian row with no deconvolution applied.
    """
    bm = gaussian_resolution_bands(WAVE, 3000.0, n_diag=41)
    s = row_sigma_kms(bm, WAVE)[50:-50]
    np.testing.assert_allclose(s, C / (2.3548 * 3000.0), rtol=5e-3)


@pytest.mark.limit
@pytest.mark.parametrize("ratio", [0.2, 0.35, 0.5])
def test_gaussian_row_deconvolves_to_quadrature_width(ratio):
    """Gaussian-in-Gaussian-out limit: deconvolving sigma_lib=ratio*sigma_R from a
    Gaussian row of width sigma_R leaves a Gaussian of width
    sigma_R * sqrt(1 - ratio^2) (quadrature subtraction of independent Gaussian
    widths, the analytic answer for R' with R' (x) G(sigma_lib) = R exactly)."""
    sig_R = C / (2.3548 * 3000.0)
    bm = gaussian_resolution_bands(WAVE, 3000.0, n_diag=41)
    out = deconvolve_library_lsf(bm, WAVE, ratio * sig_R)
    s = row_sigma_kms(out, WAVE)[50:-50]
    np.testing.assert_allclose(s, sig_R * np.sqrt(1 - ratio**2), rtol=0.02)


@pytest.mark.limit
def test_identity_when_sigma_lib_zero():
    """sigma_lib=0 -> no library LSF to remove -> R' = R exactly.

    Limit: zero deconvolution width.
    """
    bm = gaussian_resolution_bands(WAVE, 3000.0, n_diag=21)
    out = deconvolve_library_lsf(bm, WAVE, 0.0)
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(bm.data), rtol=0, atol=1e-15)


@pytest.mark.bounds
def test_refuses_library_broader_than_rows():
    """Bound: deconvolve_library_lsf must refuse sigma_lib/sigma_row > max_ratio.

    No correct model can be narrower than the library that produced it (e.g.
    MILES at ~70 km/s against a DESI z-arm row of ~25 km/s) -- the second-order
    expansion this function uses is only valid for a library width small
    against the row it is being removed from, and beyond max_ratio the
    quadrature-subtracted width is not even real (sigma_row^2 - sigma_lib^2 < 0).
    """
    bm = gaussian_resolution_bands(WAVE, 5000.0, n_diag=21)  # sigma_R ~ 25 km/s (DESI z arm)
    with pytest.raises(ValueError, match="library"):
        deconvolve_library_lsf(bm, WAVE, 70.0)  # MILES-like


@pytest.mark.conservation
def test_preserves_normalization_and_centroid_nongaussian():
    """Conservation: deconvolve_library_lsf must preserve each row's normalization
    (sum over the band = 1, i.e. total transmitted flux) exactly by construction
    (explicit renormalization), and its centroid (first moment, i.e. the row's
    mean wavelength/velocity offset) to within the second-order expansion's
    truncation error, even for a non-Gaussian (skewed, two-component) row shape.
    """
    rng = np.random.default_rng(1)
    offs = np.arange(-5, 6)
    # Skewed, two-component row shape (non-Gaussian).
    base = np.exp(-0.5 * (offs / 1.4) ** 2) + 0.15 * np.exp(-0.5 * ((offs - 2) / 1.0) ** 2)
    noise = 1 + 0.02 * rng.standard_normal((offs.size, WAVE.size))
    data = np.tile(base[:, None], (1, WAVE.size)) * noise
    data /= data.sum(axis=0, keepdims=True)
    bm = BandedMatrix(offsets=offs, data=data)
    out = deconvolve_library_lsf(bm, WAVE, 13.0)
    d = np.asarray(out.data)
    np.testing.assert_allclose(d.sum(axis=0), 1.0, atol=1e-12)
    c_in = (offs[:, None] * data).sum(0)
    c_out = (offs[:, None] * d).sum(0)
    np.testing.assert_allclose(c_out, c_in, atol=2e-3)


@pytest.mark.bounds
def test_deconvolve_handles_zero_sum_rows():
    """Bound: deconvolve_library_lsf output must be finite everywhere, even
    with zero-sum rows (masked/edge pixels, which real DESI resolution
    matrices carry). A zero-sum row has no LSF to deconvolve, so it must stay
    exactly zero (not NaN from a 0/0 renormalization) -- and because each
    row's second-order correction and renormalization only touch that row's
    own column, zeroing the edges must not perturb the interior rows at all.
    """
    sig_R = C / (2.3548 * 3000.0)
    bm_masked, bm_full, n_edge = _gaussian_bm_with_zeroed_edges()

    out_masked = deconvolve_library_lsf(bm_masked, WAVE, 0.3 * sig_R)
    out_full = deconvolve_library_lsf(bm_full, WAVE, 0.3 * sig_R)
    d_masked = np.asarray(out_masked.data)
    d_full = np.asarray(out_full.data)

    assert np.all(np.isfinite(d_masked))
    np.testing.assert_array_equal(d_masked[:, :n_edge], 0.0)
    np.testing.assert_array_equal(d_masked[:, -n_edge:], 0.0)
    np.testing.assert_allclose(
        d_masked[:, n_edge:-n_edge], d_full[:, n_edge:-n_edge], rtol=0, atol=0
    )


@pytest.mark.limit
def test_row_sigma_zero_sum_row_is_nan_without_warning():
    """Limit: a zero-sum row has no normalized weight, so row_sigma_kms must
    return NaN there (the width of an empty/masked row is undefined) rather
    than raising RuntimeWarning from a 0/0 division -- the zero-sum columns
    must be routed around the division entirely, not merely suppressed.
    """
    bm_masked, _, n_edge = _gaussian_bm_with_zeroed_edges()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = row_sigma_kms(bm_masked, WAVE)

    assert np.all(np.isnan(s[:n_edge]))
    assert np.all(np.isnan(s[-n_edge:]))
    assert np.all(np.isfinite(s[n_edge:-n_edge]))
