# SPDX-License-Identifier: BSD-3-Clause
"""Tests for z-table precomputation (free-redshift photometry).

Validates that:
1. Z-table interpolation matches fixed-z precomputation at grid points
2. Interpolation between grid points gives reasonable results
3. Gradients through z are finite (differentiable redshift)
4. Custom z grids work
5. Default z grid covers reasonable range
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.precompute import (
    interpolate_ztable,
    interpolate_ztable_smooth,
    precompute_photometry,
    precompute_photometry_ztable,
)
from tengri.utils.cosmology import luminosity_distance
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


# ── Tests: precompute_photometry_ztable ───────────────────────────


class TestZTablePrecomputation:
    """Z-table creation and structure."""

    def test_output_shapes(self, ssp_data, filters):
        """Output arrays have correct shapes."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10)
        assert zt.ssp_phot_table.shape == (10, 3, 20, 3)  # (n_z, n_met, n_age, n_filt)
        chex.assert_shape(zt.eff_waves_rest_table, (10, 3))
        chex.assert_shape(zt.log10_flux_scale_table, (10,))
        chex.assert_shape(zt.z_grid, (10,))
        assert zt.n_filters == 3

    def test_custom_z_grid(self, ssp_data, filters):
        """Custom z_grid is respected."""
        fw, ft = filters
        z_grid = jnp.array([0.01, 0.05, 0.1, 0.5, 1.0])
        zt = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid)
        assert_allclose(zt.z_grid, z_grid)
        assert zt.ssp_phot_table.shape[0] == 5

    def test_default_z_range(self, ssp_data, filters):
        """Default grid covers 0.001 to 3.0."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=50)
        assert float(zt.z_grid[0]) == pytest.approx(0.001, abs=1e-6)
        assert float(zt.z_grid[-1]) == pytest.approx(3.0, abs=1e-6)

    def test_flux_scale_decreases_with_z(self, ssp_data, filters):
        """Flux scale decreases with redshift (farther = fainter)."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)
        # flux_scale = (1+z)/(4π dL²) — dL grows faster than (1+z)
        # So flux_scale should generally decrease. Stored as log10 (#1859), and
        # log10 is monotone, so the ordering assertion is unchanged.
        assert float(zt.log10_flux_scale_table[0]) > float(zt.log10_flux_scale_table[-1])

    def test_all_values_finite(self, ssp_data, filters):
        """All precomputed values are finite."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10)
        chex.assert_tree_all_finite(zt.ssp_phot_table)
        chex.assert_tree_all_finite(zt.eff_waves_rest_table)
        chex.assert_tree_all_finite(zt.log10_flux_scale_table)


# ── Tests: interpolation accuracy ─────────────────────────────────


class TestZTableInterpolationAccuracy:
    """Z-table interpolation matches fixed-z precomputation."""

    def test_matches_fixed_z_at_grid_point(self, ssp_data, filters):
        """At a z grid point, z-table matches fixed-z precomputation exactly."""
        fw, ft = filters
        z_test = 0.1

        # Z-table with z_test on the grid
        z_grid = jnp.array([0.05, 0.1, 0.2, 0.5])
        zt = precompute_photometry_ztable(ssp_data, fw, ft, z_grid=z_grid)

        # Fixed-z precomputation
        dl_cm = luminosity_distance(z_test)
        fixed = precompute_photometry(ssp_data, fw, ft, z_test, dl_cm)

        # Interpolate z-table
        ssp_phot_interp, eff_rest_interp, log10_flux_scale_interp = interpolate_ztable(
            zt.ssp_phot_table,
            zt.eff_waves_rest_table,
            zt.log10_flux_scale_table,
            zt.z_grid,
            z_test,
        )

        assert_allclose(ssp_phot_interp, fixed.ssp_phot, rtol=1e-4)
        assert_allclose(eff_rest_interp, fixed.effective_wavelengths_rest, rtol=1e-4)
        # Compared in the LINEAR domain deliberately: rtol on a ~-57 dex log
        # would be ~1e4 times weaker than the original assertion (#1859).
        assert_allclose(10.0**log10_flux_scale_interp, 10.0**fixed.log10_flux_scale, rtol=1e-4)

    def test_interpolation_between_grid_points(self, ssp_data, filters):
        """Interpolation between grid points is within 15% of exact.

        The SSP broadband fluxes change nonlinearly with z (wavelength shift
        moves spectral features in/out of filter bandpass). With 200 z-points
        across 0.001-3.0, max interpolation error is ~5%; with 50 points, ~15%.
        For real data, use n_z=200+ or a dense custom grid around your z range.
        """
        fw, ft = filters
        z_test = 0.15  # between grid points

        # Dense z-table
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=200)

        # Exact at z_test
        dl_cm = luminosity_distance(z_test)
        fixed = precompute_photometry(ssp_data, fw, ft, z_test, dl_cm)

        # Interpolated
        ssp_phot_interp, _, log10_flux_scale_interp = interpolate_ztable(
            zt.ssp_phot_table,
            zt.eff_waves_rest_table,
            zt.log10_flux_scale_table,
            zt.z_grid,
            z_test,
        )

        # SSP photometry should agree within 15% (nonlinear z-dependence)
        frac_err = jnp.abs(ssp_phot_interp - fixed.ssp_phot) / jnp.maximum(
            jnp.abs(fixed.ssp_phot), 1e-30
        )
        assert float(jnp.max(frac_err)) < 0.15, (
            f"Max SSP photometry error: {float(jnp.max(frac_err)):.4f}"
        )

        # Flux scale should agree within 1% (smooth function of z). Compared in
        # the linear domain so the 1% keeps meaning 1% of the flux scale, not 1%
        # of its logarithm (#1859).
        fs_exact = 10.0 ** float(fixed.log10_flux_scale)
        fs_err = abs(10.0 ** float(log10_flux_scale_interp) - fs_exact) / abs(fs_exact)
        assert fs_err < 0.01, f"Flux scale error: {fs_err:.4f}"


# ── Tests: gradients ──────────────────────────────────────────────


class TestZTableGradients:
    """Gradients through z-table interpolation are finite."""

    def test_gradient_wrt_z_finite(self, ssp_data, filters):
        """Gradient w.r.t. redshift is finite (z is differentiable)."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)

        def loss(z):
            ssp_phot, eff_rest, flux_scale = interpolate_ztable(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.log10_flux_scale_table,
                zt.z_grid,
                z,
            )
            return jnp.sum(ssp_phot) + jnp.sum(eff_rest) + flux_scale

        g_jax = float(jax.grad(loss)(0.5))
        g_fd = fd_grad(loss, 0.5)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )

    def test_gradient_nonzero(self, ssp_data, filters):
        """Gradient w.r.t. z is nonzero (z actually affects the output)."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)

        def loss(z):
            ssp_phot, eff_rest, _ = interpolate_ztable(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.log10_flux_scale_table,
                zt.z_grid,
                z,
            )
            return jnp.sum(ssp_phot) + jnp.sum(eff_rest)

        g = assert_grad_matches_fd(loss, 0.5)
        assert abs(float(g)) > 1e-10, f"Gradient w.r.t. z is too small: {float(g)}"

    def test_jit_compatible(self, ssp_data, filters):
        """Z-table interpolation works inside jax.jit."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10)

        @jax.jit
        def fn(z):
            return interpolate_ztable(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.log10_flux_scale_table,
                zt.z_grid,
                z,
            )

        ssp_phot, _eff_rest, _flux_scale = fn(0.5)
        chex.assert_tree_all_finite(ssp_phot)


# ── Tests: smooth (triweight) interpolation ───────────────────────


class TestZTableSmoothInterpolation:
    """interpolate_ztable_smooth: C²-continuous triweight kernel for gradient-based inference."""

    @staticmethod
    def _scatter(zt) -> float:
        return 1.5 * float(zt.z_grid[1] - zt.z_grid[0])

    def test_output_shapes(self, ssp_data, filters):
        """Smooth interpolation returns the same shapes as interpolate_ztable."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)
        ssp_phot, eff_rest, flux_scale = interpolate_ztable_smooth(
            zt.ssp_phot_table,
            zt.eff_waves_rest_table,
            zt.log10_flux_scale_table,
            zt.z_grid,
            0.5,
            self._scatter(zt),
        )
        chex.assert_shape(ssp_phot, (3, 20, 3))
        chex.assert_shape(eff_rest, (3,))
        assert jnp.ndim(flux_scale) == 0

    def test_all_values_finite(self, ssp_data, filters):
        """Smooth interpolation returns only finite values."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)
        ssp_phot, eff_rest, flux_scale = interpolate_ztable_smooth(
            zt.ssp_phot_table,
            zt.eff_waves_rest_table,
            zt.log10_flux_scale_table,
            zt.z_grid,
            0.5,
            self._scatter(zt),
        )
        chex.assert_tree_all_finite(ssp_phot)
        chex.assert_tree_all_finite(eff_rest)
        assert jnp.isfinite(flux_scale)

    def test_gradient_wrt_z_finite(self, ssp_data, filters):
        """Gradient w.r.t. z is finite (z is differentiable through the kernel)."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)
        scatter = self._scatter(zt)

        def loss(z):
            ssp_phot, eff_rest, flux_scale = interpolate_ztable_smooth(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.log10_flux_scale_table,
                zt.z_grid,
                z,
                scatter,
            )
            return jnp.sum(ssp_phot) + jnp.sum(eff_rest) + flux_scale

        grad_jax = float(jax.grad(loss)(0.5))
        grad_fd = fd_grad(loss, 0.5)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=5e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )

    def test_gradient_smoother_than_linear(self, ssp_data, filters):
        """d(flux)/dz has lower variation than piecewise-linear across grid nodes.

        Linear interpolation has discontinuous first derivative at every grid node —
        each boundary appears as a sudden jump in d(flux)/dz. The triweight kernel
        spreads weight smoothly across neighbors (C²), so the gradient signal
        should be much quieter (lower std of consecutive differences).
        """
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)
        scatter = self._scatter(zt)

        # Dense z sweep covering 6 full grid intervals
        z_vals = jnp.linspace(float(zt.z_grid[2]), float(zt.z_grid[8]), 80)

        def loss_linear(z):
            ssp_phot, _, _ = interpolate_ztable(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.log10_flux_scale_table,
                zt.z_grid,
                z,
            )
            return jnp.sum(ssp_phot)

        def loss_smooth(z):
            ssp_phot, _, _ = interpolate_ztable_smooth(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.log10_flux_scale_table,
                zt.z_grid,
                z,
                scatter,
            )
            return jnp.sum(ssp_phot)

        grad_linear = jax.vmap(jax.grad(loss_linear))(z_vals)
        grad_smooth = jax.vmap(jax.grad(loss_smooth))(z_vals)

        variation_linear = float(jnp.std(jnp.diff(grad_linear)))
        variation_smooth = float(jnp.std(jnp.diff(grad_smooth)))
        assert variation_smooth < variation_linear, (
            f"Smooth variation ({variation_smooth:.4f}) should be less than "
            f"linear ({variation_linear:.4f})"
        )

    def test_accuracy_within_tolerance(self, ssp_data, filters):
        """Smooth interpolation stays within a reasonable error bound.

        The triweight kernel deliberately spreads weight across ~3 neighboring
        grid nodes (scatter = 1.5 × dz) to achieve C²-continuous gradients.
        This blurring trades some pointwise accuracy for gradient smoothness.
        With n_z=200 and scatter=1.5*dz, the max error is typically < 30%.
        The key property is smooth gradients, not sub-percent accuracy —
        use a dense grid (n_z ≥ 200) in production to keep errors small.
        """
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=200)
        scatter = self._scatter(zt)

        z_test = 0.15
        dl_cm = luminosity_distance(z_test)
        fixed = precompute_photometry(ssp_data, fw, ft, z_test, dl_cm)

        ssp_phot, _, _ = interpolate_ztable_smooth(
            zt.ssp_phot_table,
            zt.eff_waves_rest_table,
            zt.log10_flux_scale_table,
            zt.z_grid,
            z_test,
            scatter,
        )
        frac_err = jnp.abs(ssp_phot - fixed.ssp_phot) / jnp.maximum(jnp.abs(fixed.ssp_phot), 1e-30)
        # With n_z=200 and scatter=1.5*dz, the blurring window covers ~3 nodes
        # (±4.5 Δz ≈ ±0.067 at z~0.15), so max errors < 30% are expected.
        assert float(jnp.max(frac_err)) < 0.30, f"Max error: {float(jnp.max(frac_err)):.4f}"

    def test_jit_compatible(self, ssp_data, filters):
        """Smooth interpolation works inside jax.jit."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)
        scatter = self._scatter(zt)

        @jax.jit
        def fn(z):
            return interpolate_ztable_smooth(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.log10_flux_scale_table,
                zt.z_grid,
                z,
                scatter,
            )

        ssp_phot, _eff_rest, _flux_scale = fn(0.5)
        chex.assert_tree_all_finite(ssp_phot)
