# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.utils.wavelength — panchromatic grid and interpolation."""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.wavelength import (
    RADIO_WAVE_MAX,
    XRAY_WAVE_MIN,
    interpolate_sed_to_grid,
    make_panchromatic_grid,
)
from tests._bounds import assert_non_negative

pytestmark = pytest.mark.contract

# ── make_panchromatic_grid ────────────────────────────────────────


class TestMakePanchromaticGrid:
    """Tests for wavelength grid construction."""

    @pytest.fixture()
    def ssp_wave(self):
        """Typical SSP wavelength grid (~100 to 100,000 Å)."""
        return jnp.logspace(2, 5, 500)

    def test_no_extension_returns_ssp(self, ssp_wave):
        """Both flags False → returns ssp_wave unchanged."""
        grid = make_panchromatic_grid(ssp_wave, extend_xray=False, extend_radio=False)
        np.testing.assert_array_equal(grid, ssp_wave)

    def test_xray_extension(self, ssp_wave):
        """X-ray extension prepends points below SSP minimum."""
        grid = make_panchromatic_grid(ssp_wave, extend_xray=True, extend_radio=False)
        assert float(grid[0]) < float(ssp_wave[0])
        assert float(grid[0]) >= XRAY_WAVE_MIN
        assert len(grid) > len(ssp_wave)

    def test_xray_extension_reaches_corona_cutoff(self, ssp_wave):
        """Hard X-ray edge must sample the ~300 keV corona cutoff.

        The Yang+2020 / X-CIGALE corona has an exponential cutoff at
        E_cut = 300 keV; the grid hard edge must reach that energy so the
        rollover is sampled rather than clipped at the old ~120 keV edge
        (where exp(-120/300) = 0.67, i.e. the cutoff has barely begun).
        """
        grid = make_panchromatic_grid(ssp_wave, extend_xray=True, extend_radio=False)
        hc_kev_aa = 12.398  # h*c in keV.Angstrom
        e_max_kev = hc_kev_aa / float(grid[0])
        assert e_max_kev >= 290.0, f"hard X-ray edge only reaches {e_max_kev:.0f} keV (<290)"

    def test_radio_extension(self, ssp_wave):
        """Radio extension appends points above SSP maximum."""
        grid = make_panchromatic_grid(ssp_wave, extend_xray=False, extend_radio=True)
        assert float(grid[-1]) > float(ssp_wave[-1])
        assert float(grid[-1]) <= RADIO_WAVE_MAX * 1.001  # allow float precision
        assert len(grid) > len(ssp_wave)

    def test_both_extensions(self, ssp_wave):
        """Both extensions: X-ray + radio wings."""
        grid = make_panchromatic_grid(ssp_wave, extend_xray=True, extend_radio=True)
        assert float(grid[0]) < float(ssp_wave[0])
        assert float(grid[-1]) > float(ssp_wave[-1])

    def test_sorted(self, ssp_wave):
        """Grid is sorted ascending."""
        grid = make_panchromatic_grid(ssp_wave, extend_xray=True, extend_radio=True)
        diffs = jnp.diff(grid)
        assert jnp.all(diffs > 0), "Grid must be strictly ascending"

    def test_unique(self, ssp_wave):
        """Grid has no duplicate values."""
        grid = make_panchromatic_grid(ssp_wave, extend_xray=True, extend_radio=True)
        assert len(jnp.unique(grid)) == len(grid)

    def test_ssp_points_preserved(self, ssp_wave):
        """Original SSP wavelength points are exact in the grid."""
        grid = make_panchromatic_grid(ssp_wave, extend_xray=True, extend_radio=True)
        grid_np = np.array(grid)
        ssp_np = np.array(ssp_wave)
        # Every SSP point should appear in the grid
        for w in ssp_np:
            assert w in grid_np, f"SSP wavelength {w} not preserved in grid"

    def test_n_per_decade(self, ssp_wave):
        """Higher n_per_decade gives more points."""
        grid_sparse = make_panchromatic_grid(ssp_wave, n_per_decade=5)
        grid_dense = make_panchromatic_grid(ssp_wave, n_per_decade=50)
        assert len(grid_dense) > len(grid_sparse)


# ── interpolate_sed_to_grid ───────────────────────────────────────


class TestInterpolateSedToGrid:
    """Tests for log-log SED interpolation."""

    def test_identity(self):
        """Interpolating to same grid returns same values."""
        wave = jnp.logspace(2, 5, 100)
        sed = jnp.ones(100) * 1e8
        result = interpolate_sed_to_grid(wave, sed, wave)
        np.testing.assert_allclose(result, sed, rtol=1e-10)

    def test_power_law_accuracy(self):
        """Log-log interp is exact for power-law spectra."""
        wave_src = jnp.logspace(2, 5, 50)
        # Power law: L_nu = A * (lambda / lambda_ref)^alpha
        alpha = -1.5
        sed_src = 1e10 * (wave_src / 5000.0) ** alpha

        wave_tgt = jnp.logspace(2, 5, 200)
        sed_tgt = interpolate_sed_to_grid(wave_src, sed_src, wave_tgt)

        expected = 1e10 * (wave_tgt / 5000.0) ** alpha
        np.testing.assert_allclose(sed_tgt, expected, rtol=0.02)

    def test_zero_outside_range(self):
        """Values outside source range are zero."""
        wave_src = jnp.logspace(3, 4, 50)
        sed_src = jnp.ones(50) * 1e8
        wave_tgt = jnp.logspace(1, 6, 200)

        result = interpolate_sed_to_grid(wave_src, sed_src, wave_tgt)

        # Outside source range should be zero
        outside_lo = wave_tgt < wave_src[0]
        outside_hi = wave_tgt > wave_src[-1]
        np.testing.assert_array_equal(result[outside_lo], 0.0)
        np.testing.assert_array_equal(result[outside_hi], 0.0)

        # Inside range should be nonzero
        inside = ~outside_lo & ~outside_hi
        assert jnp.all(result[inside] > 0)

    def test_preserves_positivity(self):
        """Positive input gives positive output (within range)."""
        wave_src = jnp.logspace(2, 5, 100)
        sed_src = jnp.abs(jnp.sin(jnp.arange(100, dtype=jnp.float64))) + 0.01

        wave_tgt = jnp.logspace(2, 5, 300)
        result = interpolate_sed_to_grid(wave_src, sed_src, wave_tgt)
        assert_non_negative(result, name="result")
