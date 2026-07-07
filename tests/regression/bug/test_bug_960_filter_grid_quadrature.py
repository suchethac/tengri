# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #960 — filter-grid photometry quadrature bias.

Bug: ``lnu_filter_integral`` (and the padded batch kernel behind
``compute_flux_density_batch``) interpolated the SED onto the *filter's*
wavelength grid and trapezoid-integrated there. Instrument filter tables are
coarse (SVO/sedpy SDSS: 25-70 A spacing) while MILES-resolution spectra carry
~1 A absorption-line structure, so the quadrature point-sampled the jagged
spectrum and biased bands by up to 3 % (SDSS g vs Prospector, 2026-07 audit).
The fix integrates on the sorted union of the SED and filter nodes with the
(smooth) transmission interpolated instead.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.photometry import (
    FilterConvention,
    compute_flux_density_batch,
    lnu_filter_integral,
    pad_filters_to_bucket,
)

pytestmark = pytest.mark.regression_bug


def _jagged_sed(n_wave: int = 4000, seed: int = 42):
    """Spectrum with unresolved absorption lines on a 1 A grid.

    Flat continuum with dense, narrow (~2 A) absorption lines — the MILES
    regime that a 50 A filter table under-samples.
    """
    rng = np.random.default_rng(seed)
    wave = np.linspace(3000.0, 7000.0, n_wave)  # ~1 A spacing
    sed = np.ones_like(wave)
    line_centers = rng.uniform(3100.0, 6900.0, size=300)
    line_depths = rng.uniform(0.2, 0.8, size=300)
    for c, d in zip(line_centers, line_depths):
        sed -= d * np.exp(-0.5 * ((wave - c) / 1.0) ** 2)
    return jnp.asarray(wave), jnp.asarray(np.clip(sed, 0.05, None))


def _coarse_filter(lo: float = 3630.0, hi: float = 5830.0, dlam: float = 50.0):
    """SDSS-g-like tophat sampled at 50 A — the sedpy sdss_g0 table spacing."""
    fw = np.arange(lo, hi + dlam, dlam)
    ft = np.ones_like(fw)
    ft[0] = ft[-1] = 0.0  # taper to zero at the edges like real curves
    return jnp.asarray(fw), jnp.asarray(ft)


def _dense_reference(wave, sed, fw, ft, convention=FilterConvention.BESSELL):
    """Ground truth: photon-count band average on a very fine common grid."""
    grid = np.linspace(float(fw[0]), float(fw[-1]), 200_001)
    L = np.interp(grid, np.asarray(wave), np.asarray(sed), left=0.0, right=0.0)
    T = np.interp(grid, np.asarray(fw), np.asarray(ft), left=0.0, right=0.0)
    if convention == FilterConvention.BESSELL:
        w = T / grid
    else:
        w = T / grid**2
    return float(np.trapezoid(L * w, grid) / np.trapezoid(w, grid))


class TestFilterGridQuadrature:
    """#960: coarse filter tables must not point-sample a structured SED."""

    def test_single_filter_matches_dense_reference(self):
        wave, sed = _jagged_sed()
        fw, ft = _coarse_filter()
        ref = _dense_reference(wave, sed, fw, ft)
        got = float(lnu_filter_integral(sed, wave, fw, ft, redshift=0.0))
        # Pre-#960 quadrature errs by ~1-3 % on this configuration; the
        # union grid is exact to interpolation error.
        assert abs(got / ref - 1.0) < 1e-3, f"band mean {got:.6e} vs reference {ref:.6e}"

    def test_energy_convention_matches_dense_reference(self):
        wave, sed = _jagged_sed(seed=7)
        fw, ft = _coarse_filter()
        ref = _dense_reference(wave, sed, fw, ft, convention=FilterConvention.ENERGY)
        got = float(
            lnu_filter_integral(
                sed, wave, fw, ft, redshift=0.0, convention=FilterConvention.ENERGY
            )
        )
        assert abs(got / ref - 1.0) < 1e-3

    def test_padded_batch_matches_single(self):
        """The zero-padded batch kernel must agree with the single-filter path."""
        wave, sed = _jagged_sed(seed=3)
        fw1, ft1 = _coarse_filter()
        fw2, ft2 = _coarse_filter(5230.0, 7230.0, 50.0)  # r-like, different length
        fw_pad, ft_pad, _, n_real = pad_filters_to_bucket([fw1, fw2], [ft1, ft2])
        dl_cm = 3.086e26  # 100 Mpc
        batch = np.asarray(compute_flux_density_batch(sed, wave, fw_pad, ft_pad, 0.1, dl_cm))[
            :n_real
        ]
        from tengri.observation.photometry import compute_flux_density

        singles = [
            float(compute_flux_density(sed, wave, fw, ft, 0.1, dl_cm))
            for fw, ft in [(fw1, ft1), (fw2, ft2)]
        ]
        np.testing.assert_allclose(batch, singles, rtol=1e-12)

    def test_redshifted_band_matches_dense_reference(self):
        """The union grid must be built in the observed frame."""
        z = 0.3
        wave, sed = _jagged_sed(seed=11)
        fw, ft = _coarse_filter(4800.0, 7500.0, 60.0)
        ref = _dense_reference(np.asarray(wave) * (1 + z), sed, fw, ft)
        got = float(lnu_filter_integral(sed, wave, fw, ft, redshift=z))
        assert abs(got / ref - 1.0) < 1e-3

    def test_narrow_filter_on_coarse_sed_grid(self):
        """Union grid also covers the inverse regime: a filter narrower than
        the local SED grid spacing must not integrate to zero."""
        wave = jnp.asarray(np.geomspace(1e4, 1e6, 200))  # coarse IR grid
        sed = jnp.ones_like(wave)
        fw = jnp.asarray(np.linspace(1.00e5, 1.02e5, 64))  # narrower than dlam
        ft = jnp.asarray(np.concatenate([[0.0], np.ones(62), [0.0]]))
        got = float(lnu_filter_integral(sed, wave, fw, ft, redshift=0.0))
        assert abs(got - 1.0) < 1e-6  # flat SED: band mean must be exactly 1
