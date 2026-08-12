# SPDX-License-Identifier: BSD-3-Clause
"""Tests for IGM transmission precomputation in the z-table path.

Validates that:
1. IGM transmission is precomputed correctly at z grid points
2. Interpolation between grid points is accurate
3. With IGM disabled, results match the no-IGM path (all ones)
4. Gradients through the z-table + IGM path are finite
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.igm import igm_transmission
from tengri.components.stellar.sps.precompute import (
    interpolate_igm_ztable,
    precompute_photometry_ztable,
)
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp_data(synthetic_ssp):
    """Alias session-scoped synthetic SSP from conftest."""
    return synthetic_ssp


@pytest.fixture(scope="module")
def filters():
    """3-band synthetic filter set."""
    waves = [
        jnp.linspace(3500.0, 4500.0, 30),
        jnp.linspace(5000.0, 6500.0, 30),
        jnp.linspace(7500.0, 9000.0, 30),
    ]
    trans = [jnp.ones(30) * 0.5 for _ in range(3)]
    return waves, trans


# ── Tests: IGM precomputation in z-table ──────────────────────────


class TestIGMPrecomputation:
    """IGM transmission is correctly precomputed on the z grid."""

    def test_igm_table_shape(self, ssp_data, filters):
        """IGM table has shape (n_z, n_filters)."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10, apply_igm=True)
        chex.assert_shape(zt.igm_trans_table, (10, 3))

    def test_igm_values_in_range(self, ssp_data, filters):
        """IGM transmission is in [0, 1]."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20, apply_igm=True)
        assert jnp.all(zt.igm_trans_table >= 0.0)
        assert jnp.all(zt.igm_trans_table <= 1.0)

    def test_igm_matches_direct_computation(self, ssp_data, filters):
        """Precomputed IGM matches direct igm_transmission() call at each z."""
        fw, ft = filters
        z_grid = jnp.array([0.1, 0.5, 1.0, 2.0])
        zt = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid, apply_igm=True)

        for zi, z_val in enumerate(z_grid):
            # Compute effective observed wavelengths at this z
            eff_waves_obs = zt.eff_waves_rest_table[zi] * (1.0 + z_val)
            expected = igm_transmission(eff_waves_obs, z_val)
            assert_allclose(
                zt.igm_trans_table[zi],
                expected,
                rtol=1e-5,
                err_msg=f"IGM mismatch at z={float(z_val):.2f}",
            )

    def test_igm_all_finite(self, ssp_data, filters):
        """All IGM values are finite."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=15, apply_igm=True)
        chex.assert_tree_all_finite(zt.igm_trans_table)

    def test_igm_unity_at_low_z_red_filters(self, ssp_data, filters):
        """At low z, red filters should have ~1.0 IGM transmission.

        IGM absorption only affects rest-frame UV (< 1216 A), so at z=0.1
        only extremely blue filters would be affected.
        """
        fw, ft = filters
        z_grid = jnp.array([0.05, 0.1])
        zt = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid, apply_igm=True)
        # The reddest filter (7500-9000 A) at z=0.1 is far from Ly-alpha
        assert_allclose(zt.igm_trans_table[:, 2], 1.0, atol=1e-10)


# ── Tests: IGM disabled (default behavior) ────────────────────────


class TestIGMDisabled:
    """With apply_igm=False, igm_trans_table is all ones."""

    def test_igm_table_all_ones_by_default(self, ssp_data, filters):
        """Default (apply_igm=False) gives igm_trans_table of all ones."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10)
        assert_allclose(zt.igm_trans_table, 1.0, atol=1e-15)

    def test_igm_table_all_ones_explicit_false(self, ssp_data, filters):
        """Explicit apply_igm=False gives igm_trans_table of all ones."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10, apply_igm=False)
        assert_allclose(zt.igm_trans_table, 1.0, atol=1e-15)

    def test_ssp_phot_unchanged_by_igm_flag(self, ssp_data, filters):
        """SSP photometry table is identical with and without IGM.

        IGM is a post-processing multiplicative factor, not baked into SSPs.
        """
        fw, ft = filters
        z_grid = jnp.array([0.1, 0.5, 1.0])
        zt_no = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid, apply_igm=False)
        zt_yes = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid, apply_igm=True)
        assert_allclose(zt_no.ssp_phot_table, zt_yes.ssp_phot_table, rtol=1e-10)


# ── Tests: IGM interpolation ──────────────────────────────────────


class TestIGMInterpolation:
    """Interpolation of IGM transmission between z grid points."""

    def test_interpolation_at_grid_point(self, ssp_data, filters):
        """At a z grid point, interpolation returns the exact value."""
        fw, ft = filters
        z_grid = jnp.array([0.1, 0.5, 1.0, 2.0])
        zt = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid, apply_igm=True)
        igm_interp = interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, 0.5)
        assert_allclose(igm_interp, zt.igm_trans_table[1], rtol=1e-6)

    def test_interpolation_between_grid_points(self, ssp_data, filters):
        """Interpolation between grid points is reasonable (bounded by neighbors)."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=200, apply_igm=True)
        z_test = 0.75
        igm_interp = interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, z_test)
        # Should be in [0, 1]
        assert jnp.all(igm_interp >= 0.0)
        assert jnp.all(igm_interp <= 1.0)

        # Compare to direct computation
        # Approximate effective obs wavelengths at z_test by interpolating
        z_c = jnp.clip(z_test, zt.z_grid[0], zt.z_grid[-1])
        idx = jnp.clip(jnp.searchsorted(zt.z_grid, z_c) - 1, 0, len(zt.z_grid) - 2)
        frac = (z_c - zt.z_grid[idx]) / (zt.z_grid[idx + 1] - zt.z_grid[idx])
        eff_rest = (1.0 - frac) * zt.eff_waves_rest_table[idx] + frac * zt.eff_waves_rest_table[
            idx + 1
        ]
        eff_obs = eff_rest * (1.0 + z_test)
        expected = igm_transmission(eff_obs, z_test)
        # Dense grid should give <5% interpolation error
        assert_allclose(igm_interp, expected, rtol=0.05)

    def test_interpolation_clamped_at_edges(self, ssp_data, filters):
        """Interpolation clamps to grid boundaries."""
        fw, ft = filters
        z_grid = jnp.array([0.1, 0.5, 1.0])
        zt = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid, apply_igm=True)
        # Below minimum
        igm_low = interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, 0.01)
        assert_allclose(igm_low, zt.igm_trans_table[0], rtol=1e-6)

        # Above maximum
        igm_high = interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, 5.0)
        assert_allclose(igm_high, zt.igm_trans_table[-1], rtol=1e-6)


# ── Tests: gradients ──────────────────────────────────────────────


class TestIGMGradients:
    """Gradients through z-table + IGM path are finite."""

    def test_gradient_wrt_z_finite(self, ssp_data, filters):
        """Gradient of IGM interpolation w.r.t. z is finite."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20, apply_igm=True)

        def loss(z):
            return jnp.sum(interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, z))

        g_jax = float(jax.grad(loss)(1.0))
        g_fd = fd_grad(loss, 1.0)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )

    def test_gradient_nonzero_at_high_z(self, ssp_data):
        """At high z where IGM matters, gradient should be nonzero.

        Use a UV filter (2000-3000 A) so that at z~3, observed Ly-alpha
        (~4860 A) overlaps the filter shifted to observed frame, causing
        IGM absorption that varies with z.
        """
        # UV filter whose effective wavelength at z~3 falls near Ly-alpha
        uv_fw = [jnp.linspace(2000.0, 3000.0, 30)]
        uv_ft = [jnp.ones(30) * 0.5]
        zt = precompute_photometry_ztable(
            ssp_data, uv_fw, uv_ft, n_z=50, z_min=0.5, z_max=5.0, apply_igm=True
        )

        def loss(z):
            return jnp.sum(interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, z))

        g = assert_grad_matches_fd(loss, 3.0)
        assert abs(float(g)) > 1e-10, f"Gradient should be nonzero at z=3: {float(g)}"

    def test_gradient_combined_ssp_and_igm(self, ssp_data, filters):
        """Gradient through combined SSP + IGM z-table interpolation is finite."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20, apply_igm=True)

        def loss(z):
            # Interpolate both SSP photometry and IGM
            z_c = jnp.clip(z, zt.z_grid[0], zt.z_grid[-1])
            idx = jnp.clip(jnp.searchsorted(zt.z_grid, z_c) - 1, 0, len(zt.z_grid) - 2)
            frac = (z_c - zt.z_grid[idx]) / (zt.z_grid[idx + 1] - zt.z_grid[idx])
            ssp_phot = (1.0 - frac) * zt.ssp_phot_table[idx] + frac * zt.ssp_phot_table[idx + 1]
            igm_trans = interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, z)
            # Simulate: sum over met/age, then apply IGM per filter
            flux = jnp.sum(ssp_phot, axis=(0, 1))  # (n_filters,)
            return jnp.sum(flux * igm_trans)

        g_jax = float(jax.grad(loss)(1.0))
        g_fd = fd_grad(loss, 1.0)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )

    def test_jit_compatible(self, ssp_data, filters):
        """IGM interpolation works inside jax.jit."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10, apply_igm=True)

        @jax.jit
        def fn(z):
            return interpolate_igm_ztable(zt.igm_trans_table, zt.z_grid, z)

        result = fn(0.5)
        chex.assert_tree_all_finite(result)
        chex.assert_shape(result, (3,))
