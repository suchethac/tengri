"""Tests for z-table precomputation (free-redshift photometry).

Validates that:
1. Z-table interpolation matches fixed-z precomputation at grid points
2. Interpolation between grid points gives reasonable results
3. Gradients through z are finite (differentiable redshift)
4. Custom z grids work
5. Default z grid covers reasonable range
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.models.sps.dsps_wrapper import SSPData
from tengri.models.sps.precompute import (
    interpolate_ztable,
    precompute_photometry,
    precompute_photometry_ztable,
)
from tengri.utils.cosmology import luminosity_distance

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic SSP + filters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ssp_data():
    """Small synthetic SSP (3 Z × 20 ages × 100 λ)."""
    key = jax.random.PRNGKey(0)
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.abs(jax.random.normal(key, (3, 20, 100))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


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


# ---------------------------------------------------------------------------
# Tests: precompute_photometry_ztable
# ---------------------------------------------------------------------------


class TestZTablePrecomputation:
    """Z-table creation and structure."""

    def test_output_shapes(self, ssp_data, filters):
        """Output arrays have correct shapes."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10)
        assert zt.ssp_phot_table.shape == (10, 3, 20, 3)  # (n_z, n_met, n_age, n_filt)
        assert zt.eff_waves_rest_table.shape == (10, 3)
        assert zt.flux_scale_table.shape == (10,)
        assert zt.z_grid.shape == (10,)
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
        # So flux_scale should generally decrease
        assert float(zt.flux_scale_table[0]) > float(zt.flux_scale_table[-1])

    def test_all_values_finite(self, ssp_data, filters):
        """All precomputed values are finite."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=10)
        assert jnp.all(jnp.isfinite(zt.ssp_phot_table))
        assert jnp.all(jnp.isfinite(zt.eff_waves_rest_table))
        assert jnp.all(jnp.isfinite(zt.flux_scale_table))


# ---------------------------------------------------------------------------
# Tests: interpolation accuracy
# ---------------------------------------------------------------------------


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
        ssp_phot_interp, eff_rest_interp, flux_scale_interp = interpolate_ztable(
            zt.ssp_phot_table,
            zt.eff_waves_rest_table,
            zt.flux_scale_table,
            zt.z_grid,
            z_test,
        )

        assert_allclose(ssp_phot_interp, fixed.ssp_phot, rtol=1e-4)
        assert_allclose(eff_rest_interp, fixed.effective_wavelengths_rest, rtol=1e-4)
        assert_allclose(flux_scale_interp, fixed.flux_scale, rtol=1e-4)

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
        ssp_phot_interp, _, flux_scale_interp = interpolate_ztable(
            zt.ssp_phot_table,
            zt.eff_waves_rest_table,
            zt.flux_scale_table,
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

        # Flux scale should agree within 1% (smooth function of z)
        fs_err = abs(float(flux_scale_interp) - fixed.flux_scale) / abs(fixed.flux_scale)
        assert fs_err < 0.01, f"Flux scale error: {fs_err:.4f}"


# ---------------------------------------------------------------------------
# Tests: gradients
# ---------------------------------------------------------------------------


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
                zt.flux_scale_table,
                zt.z_grid,
                z,
            )
            return jnp.sum(ssp_phot) + jnp.sum(eff_rest) + flux_scale

        g = jax.grad(loss)(0.5)
        assert jnp.isfinite(g), f"Gradient w.r.t. z is not finite: {g}"

    def test_gradient_nonzero(self, ssp_data, filters):
        """Gradient w.r.t. z is nonzero (z actually affects the output)."""
        fw, ft = filters
        zt = precompute_photometry_ztable(ssp_data, fw, ft, n_z=20)

        def loss(z):
            ssp_phot, eff_rest, _ = interpolate_ztable(
                zt.ssp_phot_table,
                zt.eff_waves_rest_table,
                zt.flux_scale_table,
                zt.z_grid,
                z,
            )
            return jnp.sum(ssp_phot) + jnp.sum(eff_rest)

        g = jax.grad(loss)(0.5)
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
                zt.flux_scale_table,
                zt.z_grid,
                z,
            )

        ssp_phot, _eff_rest, _flux_scale = fn(0.5)
        assert jnp.all(jnp.isfinite(ssp_phot))
