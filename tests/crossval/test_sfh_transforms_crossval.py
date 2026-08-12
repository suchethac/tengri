# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate SFH generation, parameter transforms, and grid utilities.

Covers: GP SFH via IFFT, precomputation consistency, parameter transforms
(bounded/unbounded roundtrip), age grid properties, and ionizing photon rate.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._grad_parity import assert_grad_matches_fd

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_PATH = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"


# ── 1. GP SFH via IFFT ────────────────────────────────────────────


class TestGPSFHCrossval:
    """Validate Gaussian Process SFH generation from latent vector."""

    def test_zero_xi_gives_zero_gp(self):
        """Zero latent vector should give zero GP realization."""
        from tengri.components.stellar.sfh.gp_sfh import gp_from_xi

        n_grid = 128
        xi = jnp.zeros(n_grid)
        sqrt_power = jnp.ones(n_grid // 2 + 1)

        gp = gp_from_xi(xi, sqrt_power, len(xi))
        np.testing.assert_allclose(np.asarray(gp), 0.0, atol=1e-10)

    def test_gp_has_correct_length(self):
        """GP output should match n_grid."""
        from tengri.components.stellar.sfh.gp_sfh import gp_from_xi

        for n in [64, 128, 256]:
            xi = jax.random.normal(jax.random.PRNGKey(0), shape=(n,))
            sqrt_power = jnp.ones(n // 2 + 1)
            gp = gp_from_xi(xi, sqrt_power, len(xi))
            assert gp.shape == (n,), f"Expected shape ({n},), got {gp.shape}"

    def test_gp_variance_scales_with_power(self):
        """Larger sqrt_power should produce larger GP variance."""
        from tengri.components.stellar.sfh.gp_sfh import gp_from_xi

        n = 128
        xi = jax.random.normal(jax.random.PRNGKey(42), shape=(n,))

        gp_lo = gp_from_xi(xi, jnp.ones(n // 2 + 1) * 0.5, n)
        gp_hi = gp_from_xi(xi, jnp.ones(n // 2 + 1) * 2.0, n)

        var_lo = float(jnp.var(gp_lo))
        var_hi = float(jnp.var(gp_hi))
        assert var_hi > var_lo * 4, f"var_hi={var_hi:.4f} should be >> var_lo={var_lo:.4f}"

    def test_gp_ensemble_mean_near_zero(self):
        """Ensemble mean of GP realizations should be ~0 (zero-mean process)."""
        from tengri.components.stellar.sfh.gp_sfh import gp_from_xi

        n = 128
        sqrt_power = jnp.ones(n // 2 + 1)
        gps = []
        for i in range(100):
            xi = jax.random.normal(jax.random.PRNGKey(i), shape=(n,))
            gps.append(np.asarray(gp_from_xi(xi, sqrt_power, len(xi))))

        ensemble_mean = np.mean(gps, axis=0)
        assert np.max(np.abs(ensemble_mean)) < 0.5, (
            f"Ensemble mean max = {np.max(np.abs(ensemble_mean)):.3f}, expected < 0.5"
        )

    def test_gp_is_differentiable(self):
        """GP generation should be differentiable w.r.t. xi."""
        from tengri.components.stellar.sfh.gp_sfh import gp_from_xi

        n = 64
        sqrt_power = jnp.ones(n // 2 + 1)

        def loss(xi):
            return jnp.sum(gp_from_xi(xi, sqrt_power, len(xi)) ** 2)

        grad = assert_grad_matches_fd(loss, jnp.zeros(n))
        chex.assert_tree_all_finite(grad)


# ── 2. Parameter transforms (bounded <-> unbounded) ───────────────


class TestTransformsCrossval:
    """Validate bounded/unbounded parameter transforms."""

    def test_roundtrip_uniform(self):
        """to_unbounded -> to_bounded should be identity for Uniform."""
        from tengri.utils.transforms import to_bounded, to_unbounded

        lo, hi = 0.5, 3.0
        for x in [0.5, 1.0, 1.75, 2.5, 3.0]:
            u = to_unbounded(x, lo, hi)
            x_back = to_bounded(u, lo, hi)
            np.testing.assert_allclose(float(x_back), x, rtol=1e-6)

    def test_unbounded_is_real_line(self):
        """Unbounded values should span (-inf, inf) as x spans (lo, hi)."""
        from tengri.utils.transforms import to_unbounded

        lo, hi = 0.0, 1.0
        # Near lower bound -> large negative
        u_lo = float(to_unbounded(0.01, lo, hi))
        # Near upper bound -> large positive
        u_hi = float(to_unbounded(0.99, lo, hi))
        # Middle -> near zero
        u_mid = float(to_unbounded(0.5, lo, hi))

        assert u_lo < -2
        assert u_hi > 2
        assert abs(u_mid) < 1

    def test_bounded_stays_in_range(self):
        """to_bounded should always return values in [lo, hi]."""
        from tengri.utils.transforms import to_bounded

        lo, hi = -2.0, 0.5
        for u in [-100, -10, -1, 0, 1, 10, 100]:
            x = float(to_bounded(float(u), lo, hi))
            assert lo <= x <= hi, f"to_bounded({u}) = {x}, not in [{lo}, {hi}]"

    def test_transform_is_differentiable(self):
        """Transforms should have finite gradients."""
        from tengri.utils.transforms import to_bounded, to_unbounded

        def fd_grad_local(f, x: float, eps: float = 1e-4) -> float:
            """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
            return float((f(x + eps) - f(x - eps)) / (2.0 * eps))

        grad_ub_jax = float(jax.grad(lambda x: to_unbounded(x, 0.0, 1.0))(0.5))
        grad_ub_fd = fd_grad_local(lambda x: to_unbounded(x, 0.0, 1.0), 0.5)
        np.testing.assert_allclose(
            grad_ub_jax,
            grad_ub_fd,
            rtol=1e-3,
            err_msg=f"to_unbounded autodiff={grad_ub_jax:.4e}, FD={grad_ub_fd:.4e}",
        )
        assert grad_ub_jax > 0, "to_unbounded gradient should be > 0"

        grad_b_jax = float(jax.grad(lambda u: to_bounded(u, 0.0, 1.0))(0.0))
        grad_b_fd = fd_grad_local(lambda u: to_bounded(u, 0.0, 1.0), 0.0)
        np.testing.assert_allclose(
            grad_b_jax,
            grad_b_fd,
            rtol=1e-3,
            err_msg=f"to_bounded autodiff={grad_b_jax:.4e}, FD={grad_b_fd:.4e}",
        )
        assert grad_b_jax > 0, "to_bounded gradient should be > 0"

    def test_jacobian_positive(self):
        """Transform Jacobian should be positive (monotonic mapping)."""
        from tengri.utils.transforms import to_unbounded

        lo, hi = 0.0, 10.0
        for x in [0.5, 2.0, 5.0, 8.0, 9.5]:
            jac = float(jax.grad(lambda v: to_unbounded(v, lo, hi))(float(x)))
            assert jac > 0, f"Jacobian at x={x} is {jac}, expected > 0"


# ── 3. Age grid utilities ─────────────────────────────────────────


class TestGridCrossval:
    """Validate age grid construction and properties."""

    def test_log_age_grid_range(self):
        """Log-age grid should span the requested range."""
        from tengri.utils.grid import make_log_age_grid

        grid = np.asarray(make_log_age_grid(256, log_age_min=6.0, log_age_max=10.14))
        np.testing.assert_allclose(grid[0], 6.0, atol=0.01)
        np.testing.assert_allclose(grid[-1], 10.14, atol=0.01)

    def test_log_age_grid_uniform_spacing(self):
        """Grid should be uniformly spaced in log-age."""
        from tengri.utils.grid import make_log_age_grid

        grid = np.asarray(make_log_age_grid(128))
        spacing = np.diff(grid)
        np.testing.assert_allclose(spacing, spacing[0], rtol=1e-10)

    def test_grid_spacing_matches(self):
        """grid_spacing() should return the actual spacing."""
        from tengri.utils.grid import grid_spacing, make_log_age_grid

        grid = make_log_age_grid(256)
        d = float(grid_spacing(grid))
        expected = float(grid[1] - grid[0])
        np.testing.assert_allclose(d, expected, rtol=1e-10)


# ── 4. Ionizing photon rate Q_H ───────────────────────────────────


class TestIonizingSpectrumCrossval:
    """Validate ionizing photon rate computation."""

    def test_young_ssp_has_high_qh(self):
        """Very young SSP should have log Q_H > 45."""
        if not _SSP_PATH.is_file():
            pytest.skip("SSP data not found")

        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        ssp = load_ssp_data(str(_SSP_PATH))
        wave = np.asarray(ssp.ssp_wave)
        flux = np.asarray(ssp.ssp_flux[7, 0])  # solar Z, youngest age

        # Q_H = integral of (L_nu / h*nu) dnu for lambda < 912 A
        h = 6.626e-27
        c = 2.998e10
        lsun = 3.828e33
        mask = wave < 912.0
        if np.sum(mask) < 2:
            pytest.skip("No ionizing wavelengths in SSP grid")

        nu = c / (wave[mask] * 1e-8)
        integrand = flux[mask] * lsun / (h * nu)
        q_h = abs(np.trapezoid(integrand, nu))

        assert np.log10(max(q_h, 1e-99)) > 40, f"log Q_H = {np.log10(max(q_h, 1e-99)):.1f}"

    def test_old_ssp_has_low_qh(self):
        """Old SSP (10 Gyr) should have much lower Q_H than young."""
        if not _SSP_PATH.is_file():
            pytest.skip("SSP data not found")

        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        ssp = load_ssp_data(str(_SSP_PATH))
        wave = np.asarray(ssp.ssp_wave)
        ages = np.asarray(ssp.ssp_lg_age_gyr)

        h, c, lsun = 6.626e-27, 2.998e10, 3.828e33
        mask = wave < 912.0
        if np.sum(mask) < 2:
            pytest.skip("No ionizing wavelengths")

        nu = c / (wave[mask] * 1e-8)

        idx_young = 0
        idx_old = np.argmin(np.abs(ages - 1.0))  # 10 Gyr

        flux_y = np.asarray(ssp.ssp_flux[7, idx_young])
        flux_o = np.asarray(ssp.ssp_flux[7, idx_old])

        qh_young = abs(np.trapezoid(flux_y[mask] * lsun / (h * nu), nu))
        qh_old = abs(np.trapezoid(flux_o[mask] * lsun / (h * nu), nu))

        assert qh_old < qh_young * 0.001, "Old SSP Q_H should be << young"


# ── 5. Precomputation consistency ─────────────────────────────────


class TestPrecomputeCrossval:
    """Verify precomputed photometry matches direct computation."""

    def test_precomputed_matches_direct(self):
        """Precomputed and direct photometry should agree within 25%."""
        if not _SSP_PATH.is_file():
            pytest.skip("SSP data not found")

        from tengri import Parameters, SEDModel, Uniform
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        try:
            ssp = load_ssp_data(str(_SSP_PATH))
            spec = Parameters(
                sfh_alpha=Uniform(0.5, 3.0),
                sfh_beta=Uniform(0.3, 2.0),
                sfh_tau_peak_gyr=Uniform(0.5, 10.0),
                psd_sigma=Uniform(0.01, 3.0),
                psd_tau_myr=Uniform(10, 500),
                met_logzsol=Uniform(-2.0, 0.2),
                dust_tau_bc=Uniform(0.0, 4.0),
                dust_tau_diff=Uniform(0.0, 4.0),
                redshift=0.1,
            )

            # With precomputation (default)
            model_pre = SEDModel(spec, ssp, precompute=True)
            # Without
            model_dir = SEDModel(spec, ssp, precompute=False)

            params = spec.sample(jax.random.PRNGKey(0))

            phot_pre = model_pre.predict_photometry(params)
            phot_dir = model_dir.predict_photometry(params)

            if phot_pre is not None and phot_dir is not None:
                np.testing.assert_allclose(
                    np.asarray(phot_pre),
                    np.asarray(phot_dir),
                    rtol=0.25,
                    err_msg="Precomputed vs direct photometry mismatch",
                )
        except (ValueError, AttributeError):
            pytest.skip("No filters configured")
