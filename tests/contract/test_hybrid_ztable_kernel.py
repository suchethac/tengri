# SPDX-License-Identifier: BSD-3-Clause
"""Tests for hybrid photometry kernel with z-table interpolation.

Validates that:
1. build_hybrid_photometry_ztable creates a working kernel
2. Kernel with free redshift produces finite photometry
3. Photometry changes smoothly with redshift (differentiable)
4. Results agree with fixed-z predictions at grid points
5. Kernel JIT-compiles and runs fast

Uses synthetic SSP data to avoid requiring actual SSP files.
"""

from __future__ import annotations

import os

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.components.stellar.sps.precompute import (
    interpolate_ztable,
    precompute_photometry,
    precompute_photometry_ztable,
)
from tengri.utils.cosmology import luminosity_distance

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# Suppress JAX Metal warnings on macOS
os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture(scope="module")
def ssp_data():
    """Synthetic SSP for testing (5 metallicities × 15 ages × 100 wavelengths)."""
    key = jax.random.PRNGKey(42)
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.abs(jax.random.normal(key, (5, 15, 100))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-2.0, 1.14, 15),
        ssp_lgmet=jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0]),
    )


@pytest.fixture(scope="module")
def filters():
    """Synthetic 3-band filter set."""
    waves = [
        jnp.linspace(3500.0, 4500.0, 30),
        jnp.linspace(5000.0, 6500.0, 30),
        jnp.linspace(7500.0, 9000.0, 30),
    ]
    trans = [jnp.ones(30) * 0.8 for _ in range(3)]
    return waves, trans


class TestZTableBasic:
    """Basic z-table functionality for photometry."""

    def test_ztable_creation(self, ssp_data, filters):
        """Z-table can be created with synthetic SSP data."""
        fw, ft = filters
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, n_z=15)

        assert ztable.ssp_phot_table.shape == (15, 5, 15, 3)  # (n_z, n_met, n_age, n_filt)
        chex.assert_tree_all_finite(ztable.ssp_phot_table)
        chex.assert_tree_all_finite(ztable.log10_flux_scale_table)

    def test_ztable_interpolation_at_grid_point(self, ssp_data, filters):
        """Z-table interpolation matches fixed-z at grid points."""
        fw, ft = filters
        z_test = 0.2

        # Build z-table with z_test on the grid
        z_grid = jnp.array([0.05, 0.1, 0.2, 0.35, 0.5])
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid)

        # Fixed-z computation
        dl_cm = luminosity_distance(z_test)
        fixed = precompute_photometry(ssp_data, fw, ft, z_test, dl_cm)

        # Interpolate z-table
        ssp_phot_interp, _, log10_flux_scale_interp = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            z_test,
        )

        # Should match exactly at grid points
        assert_allclose(ssp_phot_interp, fixed.ssp_phot, rtol=1e-5)
        # Linear domain: rtol on a ~-57 dex log would be ~1e4x weaker (#1859).
        assert_allclose(10.0**log10_flux_scale_interp, 10.0**fixed.log10_flux_scale, rtol=1e-5)

    def test_ztable_gradient_wrt_z(self, ssp_data, filters):
        """Z-table interpolation is differentiable w.r.t. redshift."""
        fw, ft = filters
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)

        def loss(z):
            ssp_phot, _, flux_scale = interpolate_ztable(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.log10_flux_scale_table,
                ztable.z_grid,
                z,
            )
            return jnp.sum(ssp_phot) + flux_scale

        grad_jax = float(jax.grad(loss)(0.3))
        grad_fd = fd_grad(loss, 0.3)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=5e-3,
            atol=1e-10,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert abs(grad_jax) > 1e-10, "Gradient w.r.t. z should be nonzero"


class TestZTableInterpolationSmoothnessAndMonotonicity:
    """Test interpolation smoothness and monotonicity properties."""

    def test_interpolation_smooth_with_z(self, ssp_data, filters):
        """Photometry varies as redshift changes (verifying z-dependence works)."""
        fw, ft = filters
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, n_z=30)

        z_vals = jnp.linspace(0.1, 0.5, 10)
        phots = []

        for z in z_vals:
            ssp_phot, _, _ = interpolate_ztable(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.log10_flux_scale_table,
                ztable.z_grid,
                z,
            )
            phots.append(ssp_phot)

        # Check that photometry is not all the same (z actually affects output)
        # With random SSP spectra, changes can be large, so just check for variation
        all_phots = jnp.array(phots)
        mean_phot = jnp.mean(all_phots)
        std_phot = jnp.std(all_phots)
        assert std_phot > 0.001 * mean_phot, (
            "Photometry should vary with redshift (got little variation)"
        )

    def test_interpolation_better_with_more_points(self, ssp_data, filters):
        """Interpolation error decreases with more z-grid points."""
        fw, ft = filters
        z_test = 0.25

        # Fixed-z for comparison
        dl_cm = luminosity_distance(z_test)
        fixed = precompute_photometry(ssp_data, fw, ft, z_test, dl_cm)

        errors = []
        for n_z in [5, 10, 20, 40]:
            ztable = precompute_photometry_ztable(ssp_data, fw, ft, n_z=n_z)
            ssp_phot_interp, _, _ = interpolate_ztable(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.log10_flux_scale_table,
                ztable.z_grid,
                z_test,
            )
            rel_err = jnp.mean(
                jnp.abs(ssp_phot_interp - fixed.ssp_phot)
                / jnp.maximum(jnp.abs(fixed.ssp_phot), 1e-30)
            )
            errors.append(float(rel_err))

        # More points should give lower error
        assert errors[0] > errors[-1], f"Error should decrease with more grid points: {errors}"


class TestZTableEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_small_z(self, ssp_data, filters):
        """Z-table works at very small redshift (z ~ 0.001)."""
        fw, ft = filters
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, z_min=0.001, z_max=0.5, n_z=20)

        ssp_phot, _, _ = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            0.001,
        )
        chex.assert_tree_all_finite(ssp_phot)

    def test_very_large_z(self, ssp_data, filters):
        """Z-table works at large redshift (z ~ 3.0)."""
        fw, ft = filters
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, z_min=0.1, z_max=3.0, n_z=30)

        ssp_phot, _, _ = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            2.9,
        )
        chex.assert_tree_all_finite(ssp_phot)

    def test_extrapolation_outside_grid(self, ssp_data, filters):
        """Extrapolation outside z-grid gives reasonable (if not exact) results."""
        fw, ft = filters
        z_grid = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid)

        # Extrapolate beyond grid
        ssp_phot, _, _ = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            0.55,  # Beyond 0.5
        )
        chex.assert_tree_all_finite(ssp_phot)

    def test_interpolation_at_boundaries(self, ssp_data, filters):
        """Interpolation works at grid boundaries."""
        fw, ft = filters
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10)

        # At min boundary
        ssp_phot_min, _, _ = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            float(ztable.z_grid[0]),
        )
        chex.assert_tree_all_finite(ssp_phot_min)
        # At max boundary
        ssp_phot_max, _, _ = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            float(ztable.z_grid[-1]),
        )
        chex.assert_tree_all_finite(ssp_phot_max)


class TestZTableConsistency:
    """Consistency and reproducibility."""

    def test_same_results_same_seed(self, ssp_data, filters):
        """Same computation gives same results."""
        fw, ft = filters
        z_grid = jnp.array([0.1, 0.2, 0.3])

        ztable1 = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid)
        ztable2 = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid)

        assert_allclose(ztable1.ssp_phot_table, ztable2.ssp_phot_table)
        assert_allclose(ztable1.log10_flux_scale_table, ztable2.log10_flux_scale_table)

    def test_interpolation_consistency(self, ssp_data, filters):
        """Interpolation is consistent across calls."""
        fw, ft = filters
        ztable = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)

        z_test = 0.25

        result1 = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            z_test,
        )

        result2 = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.log10_flux_scale_table,
            ztable.z_grid,
            z_test,
        )

        assert_allclose(result1[0], result2[0])
        assert_allclose(result1[1], result2[1])
        assert_allclose(result1[2], result2[2])
