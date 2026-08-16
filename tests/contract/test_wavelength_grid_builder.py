# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for build_wavelength_grid."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectroscopy import build_wavelength_grid

pytestmark = pytest.mark.contract


def _constant_r(wave_um):
    """Constant R=100 for testing."""
    return 100.0 * jnp.ones_like(wave_um)


class TestBuildWavelengthGrid:
    """Wavelength grid builder produces correctly spaced grids."""

    def test_constant_r_spacing(self):
        """For constant R, pixel spacing should be proportional to wavelength."""
        grid = build_wavelength_grid(_constant_r, 10000.0, 50000.0, n_pix_per_resel=2.5)
        diffs = np.diff(np.asarray(grid))
        ratios = diffs / np.asarray(grid[:-1])
        expected = 1.0 / (2.5 * 100.0)
        np.testing.assert_allclose(ratios, expected, rtol=0.01)

    def test_covers_range(self):
        """Grid starts at wave_min and ends at or before wave_max."""
        grid = build_wavelength_grid(_constant_r, 10000.0, 50000.0)
        assert float(grid[0]) == 10000.0
        assert float(grid[-1]) <= 50000.0

    def test_monotonically_increasing(self):
        """Grid wavelengths are strictly increasing."""
        grid = build_wavelength_grid(_constant_r, 10000.0, 50000.0)
        assert np.all(np.diff(np.asarray(grid)) > 0.0)

    def test_nirspec_prism_fractional_spacing(self):
        """NIRSpec PRISM: fractional spacing dλ/λ is larger at blue end (lower R)."""
        from tengri.observation.spectrum import nirspec_prism_resolution

        grid = build_wavelength_grid(nirspec_prism_resolution, 6000.0, 53000.0)
        g = np.asarray(grid)
        frac = np.diff(g) / g[:-1]
        n = len(frac)
        blue_frac = np.mean(frac[: n // 4])
        red_frac = np.mean(frac[3 * n // 4 :])
        assert blue_frac > red_frac, (
            f"Blue fractional spacing ({blue_frac:.5f}) should be larger than "
            f"red ({red_frac:.5f}) for PRISM (lower R at blue end)"
        )

    def test_n_pix_per_resel_affects_density(self):
        """More pixels per resolution element → denser grid."""
        grid_coarse = build_wavelength_grid(_constant_r, 10000.0, 50000.0, n_pix_per_resel=2.0)
        grid_fine = build_wavelength_grid(_constant_r, 10000.0, 50000.0, n_pix_per_resel=4.0)
        assert grid_fine.shape[0] > grid_coarse.shape[0]
