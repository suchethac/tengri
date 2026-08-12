# SPDX-License-Identifier: BSD-3-Clause
"""Tests for K&D 2018 3-zone disc preintegration.

Verifies that the preintegrated (filter-level) K&D computation matches
the full-wavelength kubota_done_disc() to within acceptable photometric
tolerance. The preintegration approximation errors come from:

1. L_bol estimation via filter bandwidths (few %)
2. Filter-level vs wavelength-level Planck integration (sub-%)
3. nthcomp interpolation on the filter grid vs frequency grid (<1%)

Target: <5% per-filter error for SDSS ugriz at z=0.1.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tests._bounds import assert_non_negative


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Helpers: synthetic filters ────────────────────────────────────
def _make_tophat_filter(center_aa: float, width_aa: float, n_points: int = 100):
    """Create a tophat filter transmission curve."""
    wave = np.linspace(center_aa - width_aa / 2, center_aa + width_aa / 2, n_points)
    trans = np.ones_like(wave)
    return wave, trans


def _make_sdss_like_filters():
    """Approximate SDSS ugriz filter set."""
    centers = [3551, 4686, 6166, 7480, 8932]  # Angstrom
    widths = [560, 1390, 1370, 1510, 1170]
    filter_waves = []
    filter_trans = []
    for c, w in zip(centers, widths):
        fw, ft = _make_tophat_filter(c, w, n_points=200)
        filter_waves.append(fw)
        filter_trans.append(ft)
    return filter_waves, filter_trans


# ── Test: Planck filter table ─────────────────────────────────────
class TestPlanckFilterTable:
    """Test that filter-integrated Planck matches direct computation."""

    def test_planck_table_shape(self):
        from tengri.components.agn.kd_precompute import _build_planck_filter_table

        T_grid = np.geomspace(1000, 1e6, 50)
        fw, ft = _make_sdss_like_filters()
        table = _build_planck_filter_table(T_grid, fw, ft, redshift=0.1)
        chex.assert_shape(table, (50, 5))
        assert_non_negative(table, name="table")

    def test_planck_table_monotonic_in_T(self):
        """Hotter temperatures should give more flux in bluer filters."""
        from tengri.components.agn.kd_precompute import _build_planck_filter_table

        T_grid = np.geomspace(1e4, 1e6, 100)
        fw, ft = _make_sdss_like_filters()
        table = _build_planck_filter_table(T_grid, fw, ft, redshift=0.0)
        # u-band should increase with T over this range
        assert np.all(np.diff(table[:, 0]) > 0), "u-band should increase with T"

    def test_planck_lookup_matches_direct(self):
        """Lookup at exact grid temperatures should match table values."""
        from tengri.components.agn.kd_precompute import (
            _build_planck_filter_table,
            _lookup_planck_filter,
        )

        T_grid = np.geomspace(1000, 1e6, 100)
        fw, ft = _make_sdss_like_filters()
        table = _build_planck_filter_table(T_grid, fw, ft, redshift=0.0)
        # At an exact grid point
        T_test = T_grid[50]
        result = _lookup_planck_filter(T_test, jnp.array(T_grid), jnp.array(table))
        expected = table[50]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_planck_lookup_interpolation(self):
        """Lookup between grid points should be smooth."""
        from tengri.components.agn.kd_precompute import (
            _build_planck_filter_table,
            _lookup_planck_filter,
        )

        T_grid = np.geomspace(1000, 1e6, 200)
        fw, ft = _make_sdss_like_filters()
        table = _build_planck_filter_table(T_grid, fw, ft, redshift=0.0)
        T_grid_jax = jnp.array(T_grid)
        table_jax = jnp.array(table)
        # Interpolate at midpoint between grid points
        T_mid = np.sqrt(T_grid[50] * T_grid[51])  # geometric midpoint
        result = _lookup_planck_filter(T_mid, T_grid_jax, table_jax)
        # Should be between the two grid values
        lo = table[50]
        hi = table[51]
        for f in range(5):
            assert (
                lo[f] <= result[f] + 1e-30 <= hi[f] + 1e-30
                or hi[f] <= result[f] + 1e-30 <= lo[f] + 1e-30
            )


# ── Test: Corona filter table ─────────────────────────────────────
class TestCoronaFilterTable:
    """Test the hot corona preintegration."""

    def test_corona_table_shape(self):
        from tengri.components.agn.kd_precompute import _build_corona_filter_table

        Gamma_grid = np.linspace(1.4, 3.0, 10)
        kT_grid = np.geomspace(10, 500, 8)
        kTbb_grid = np.geomspace(5e-4, 0.12, 6)  # seed-photon temperature axis
        fw, ft = _make_sdss_like_filters()
        table = _build_corona_filter_table(Gamma_grid, kT_grid, kTbb_grid, fw, ft, redshift=0.1)
        chex.assert_shape(table, (10, 8, 6, 5))

    def test_corona_harder_gives_less_optical(self):
        """Steeper Gamma (softer) should give MORE optical flux."""
        from tengri.components.agn.kd_precompute import _build_corona_filter_table

        Gamma_grid = np.linspace(1.4, 3.0, 20)
        kT_grid = np.array([100.0])
        # Smallest seed temperature: rollover knee sits in the IR, leaving the
        # optical r-band (well above the knee) governed by the power-law slope.
        kTbb_grid = np.array([5e-4])
        fw, ft = _make_sdss_like_filters()
        table = _build_corona_filter_table(Gamma_grid, kT_grid, kTbb_grid, fw, ft, redshift=0.0)
        # Shape (20, 1, 1, 5) — r-band (index 2), seed index 0
        r_band = table[:, 0, 0, 2]
        # Softer (larger Gamma) = more steep = more UV/optical photons
        # For fixed kT, steeper power law puts more flux at lower frequencies
        # This should generally increase (though depends on normalization)
        assert r_band[-1] > r_band[0], "Softer Gamma should give more optical"

    def test_corona_seed_rollover_suppresses_optical(self):
        """A higher seed temperature shifts the rollover blueward, removing optical flux.

        The hot corona is a thermal-Comptonization spectrum bounded below by its
        seed-photon energy (K&D 2018, Section 2.2). Raising the seed temperature
        moves the low-energy rollover toward the UV, so the optical r-band — now
        below the knee — must lose flux relative to a low (IR-knee) seed.
        """
        from tengri.components.agn.kd_precompute import _build_corona_filter_table

        Gamma_grid = np.array([1.8])
        kT_grid = np.array([100.0])
        kTbb_grid = np.array([5e-4, 0.12])  # IR knee vs EUV knee
        fw, ft = _make_sdss_like_filters()
        table = _build_corona_filter_table(Gamma_grid, kT_grid, kTbb_grid, fw, ft, redshift=0.0)
        r_band_low_seed = table[0, 0, 0, 2]
        r_band_high_seed = table[0, 0, 1, 2]
        assert r_band_high_seed < r_band_low_seed, (
            "Higher seed temperature must suppress optical corona flux"
        )


# ── Test: nthcomp filter table ────────────────────────────────────
class TestNthcompFilterTable:
    """Test nthcomp preintegration (requires templates)."""

    def test_nthcomp_table_loads(self):
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates not available")
        from tengri.components.agn.kd_precompute import _build_nthcomp_filter_table

        fw, ft = _make_sdss_like_filters()
        table, _gamma, _kTe, _kTbb = _build_nthcomp_filter_table(fw, ft, redshift=0.1)
        assert table is not None
        assert table.shape == (20, 15, 50, 5)  # (gamma, kTe, kTbb, filters)
        assert_non_negative(table, name="table")

    def test_nthcomp_lookup_smooth(self):
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates not available")
        from tengri.components.agn.kd_precompute import (
            _build_nthcomp_filter_table,
            _lookup_nthcomp_filter,
        )

        fw, ft = _make_sdss_like_filters()
        table, gamma_grid, kTe_grid, kTbb_grid = _build_nthcomp_filter_table(fw, ft, redshift=0.1)
        table_jax = jnp.array(table)
        gamma_jax = jnp.array(gamma_grid)
        kTe_jax = jnp.array(kTe_grid)
        kTbb_jax = jnp.array(kTbb_grid)
        # Query at grid center
        result = _lookup_nthcomp_filter(
            gamma_grid[10],
            kTe_grid[7],
            kTbb_grid[25],
            gamma_jax,
            kTe_jax,
            kTbb_jax,
            table_jax,
        )
        expected = table[10, 7, 25]
        np.testing.assert_allclose(result, expected, rtol=1e-4)


# ── Test: Full preintegration pipeline ────────────────────────────
class TestKDPreintegrationPipeline:
    """End-to-end test comparing preintegrated vs full-wavelength K&D."""

    def test_preintegrate_builds(self):
        from tengri.components.agn.kd_precompute import preintegrate_kd_components

        fw, ft = _make_sdss_like_filters()
        kd = preintegrate_kd_components(fw, ft, redshift=0.1)
        assert kd.n_filters == 5
        assert kd.planck_table.shape[1] == 5
        assert kd.corona_table.ndim == 4  # (Gamma, kT, kTbb_seed, filters)
        assert kd.corona_table.shape[3] == 5

    def test_preintegrated_vs_full_wavelength(self):
        """Compare preintegrated K&D against full-wavelength computation.
        This is the key accuracy test. Target: <10% per-filter error
        for typical AGN parameters at z=0.1 with SDSS-like filters.
        """
        from tengri.components.agn.disc import kubota_done_disc
        from tengri.components.agn.kd_precompute import (
            kubota_done_disc_preintegrated,
            preintegrate_kd_components,
        )

        z = 0.1
        fw, ft = _make_sdss_like_filters()
        # Reference: full-wavelength K&D — use RELAGN's grid range [1e-4, 1e4] keV
        # = [0.124, 1.24e5] Angstrom. The corona normalization uses the same
        # fixed grid internally, so results are consistent when the overall
        # SED normalization also spans this range.
        wave_rest = np.geomspace(0.124, 1.24e5, 20000)
        # Build precomputed tables
        kd_data = preintegrate_kd_components(fw, ft, redshift=z, n_T=300)
        params = dict(
            agn_log_lbol=11.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_a_spin=0.0,
            agn_cos_inc=0.5,
            agn_f_hard=0.02,
            agn_gamma_warm=2.5,
            agn_kt_warm=0.2,
            agn_gamma_hard=1.8,
            agn_kt_hot=100.0,
            agn_r_warm_ratio=2.0,
        )
        l_nu_full = kubota_done_disc(jnp.array(wave_rest), **params)
        # Compute photometry from full wavelength SED
        wave_obs = wave_rest * (1.0 + z)
        full_phot = np.zeros(5)
        for f_idx, (fw_i, ft_i) in enumerate(zip(fw, ft)):
            fw_np = np.asarray(fw_i)
            ft_np = np.asarray(ft_i)
            # Interpolate SED onto filter grid
            sed_on_filt = np.interp(fw_np, wave_obs, np.array(l_nu_full), left=0, right=0)
            num = _np_trapezoid(sed_on_filt * ft_np * fw_np, fw_np)
            denom = _np_trapezoid(ft_np * fw_np, fw_np)
            full_phot[f_idx] = num / max(denom, 1e-30)
        # Preintegrated path
        preint_phot = kubota_done_disc_preintegrated(kd_data, **params)
        # Compare (allow up to 10% per filter)
        for f_idx in range(5):
            if full_phot[f_idx] > 0:
                rel_err = abs(preint_phot[f_idx] - full_phot[f_idx]) / full_phot[f_idx]
                assert rel_err < 0.10, (
                    f"Filter {f_idx}: preint={preint_phot[f_idx]:.4e}, "
                    f"full={full_phot[f_idx]:.4e}, err={rel_err:.1%}"
                )

    def test_preintegrated_gradient_finite(self):
        """Gradients of preintegrated K&D should be finite."""
        from tengri.components.agn.kd_precompute import (
            kubota_done_disc_preintegrated,
            preintegrate_kd_components,
        )

        fw, ft = _make_sdss_like_filters()
        kd_data = preintegrate_kd_components(fw, ft, redshift=0.1)

        def _loss(log_lbol):
            phot = kubota_done_disc_preintegrated(
                kd_data,
                agn_log_lbol=log_lbol,
                agn_log_mbh=8.0,
                agn_log_ledd=-1.0,
            )
            return jnp.sum(phot)

        grad_jax = float(jax.grad(_loss)(11.0))
        grad_fd = fd_grad(_loss, 11.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax != 0.0, "Gradient is zero (no sensitivity)"

    def test_preintegrated_jit_compatible(self):
        """Preintegrated K&D should work under JIT."""
        from tengri.components.agn.kd_precompute import (
            kubota_done_disc_preintegrated,
            preintegrate_kd_components,
        )

        fw, ft = _make_sdss_like_filters()
        kd_data = preintegrate_kd_components(fw, ft, redshift=0.1)

        @jax.jit
        def _predict(log_lbol):
            return kubota_done_disc_preintegrated(
                kd_data,
                agn_log_lbol=log_lbol,
            )

        result = _predict(11.0)
        chex.assert_shape(result, (5,))
        chex.assert_tree_all_finite(result)


# numpy >= 2.0 uses trapezoid; older versions used trapz
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
