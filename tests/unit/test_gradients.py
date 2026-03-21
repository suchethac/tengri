"""Tests that gradients are meaningful (not just finite).

Verifies that:
1. Gradients point in physically sensible directions
2. Autodiff gradients match finite differences
3. Gradient magnitudes are reasonable (not vanishing or exploding)
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.models.dust.attenuation import two_component_dust
from tengri.models.sfh.gp_sfh import compute_sqrt_power_drw, gp_from_xi
from tengri.models.sfh.mean_sfh import double_powerlaw
from tengri.models.sfh.psd_models import drw_variance
from tengri.utils.grid import grid_spacing, log_age_to_age_yr, make_log_age_grid

jax.config.update("jax_enable_x64", True)

N_GRID = 256


class TestGradientFiniteDifference:
    """Autodiff vs finite-difference comparison (T5 from roadmap)."""

    @pytest.fixture
    def setup(self):
        log_age_grid = make_log_age_grid(N_GRID)
        d = grid_spacing(log_age_grid)
        sqrt_power = compute_sqrt_power_drw(N_GRID, float(d), 1.0, 50e6)
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))
        return sqrt_power, xi

    def test_gp_grad_vs_finite_diff(self, setup):
        """GP gradient w.r.t. xi matches finite differences."""
        sqrt_power, xi = setup

        def f(xi):
            return jnp.sum(gp_from_xi(xi, sqrt_power, N_GRID) ** 2)

        # Autodiff
        grad_auto = jax.grad(f)(xi)

        # Finite difference (central)
        eps = 1e-5
        grad_fd = jnp.zeros_like(xi)
        for i in range(min(10, N_GRID)):  # Check first 10 components
            xi_plus = xi.at[i].add(eps)
            xi_minus = xi.at[i].add(-eps)
            grad_fd_i = (f(xi_plus) - f(xi_minus)) / (2 * eps)
            assert_allclose(
                float(grad_auto[i]),
                float(grad_fd_i),
                rtol=1e-3,
                err_msg=f"Gradient mismatch at xi[{i}]",
            )

    def test_dust_grad_vs_finite_diff(self):
        """Dust attenuation gradient matches finite differences."""
        wave = jnp.linspace(3000, 8000, 50)
        ages = jnp.logspace(6, 10, 30)

        def f(tau_v1):
            return jnp.sum(
                two_component_dust(
                    wave, ages, tau_v1, 0.3, law_bc="power_law", law_diff="power_law"
                )
            )

        grad_auto = jax.grad(f)(0.5)

        eps = 1e-6
        grad_fd = (f(0.5 + eps) - f(0.5 - eps)) / (2 * eps)
        assert_allclose(float(grad_auto), float(grad_fd), rtol=1e-3)

    def test_mean_sfh_grad_vs_finite_diff(self):
        """Double power law gradient matches finite differences."""
        t = jnp.logspace(7, 10, 100)

        def f(alpha):
            return jnp.sum(double_powerlaw(t, alpha, 1.0, 1e9, 10.0))

        grad_auto = jax.grad(f)(1.5)

        eps = 1e-6
        grad_fd = (f(1.5 + eps) - f(1.5 - eps)) / (2 * eps)
        assert_allclose(float(grad_auto), float(grad_fd), rtol=1e-3)


class TestGradientDirection:
    """Verify gradients point in physically correct directions."""

    def test_more_dust_decreases_flux(self):
        """d(total_flux)/d(tau_v1) < 0: more dust = less flux."""
        wave = jnp.linspace(3000, 8000, 50)
        ages = jnp.logspace(6, 10, 30)

        def total_flux(tau_v1):
            return jnp.sum(
                two_component_dust(
                    wave, ages, tau_v1, 0.3, law_bc="power_law", law_diff="power_law"
                )
            )

        grad = jax.grad(total_flux)(0.5)
        assert float(grad) < 0, "More dust should decrease total flux"

    def test_higher_sfr_norm_increases_sfr(self):
        """d(total_SFR)/d(norm) > 0."""
        t = jnp.logspace(7, 10, 100)

        def total_sfr(norm):
            return jnp.sum(double_powerlaw(t, 1.0, 1.0, 1e9, norm))

        grad = jax.grad(total_sfr)(10.0)
        assert float(grad) > 0, "Higher norm should increase total SFR"

    def test_gp_xi_gradient_not_vanishing(self):
        """GP gradients w.r.t. xi are not vanishing."""
        d = (10.14 - 6.0) / (N_GRID - 1)
        sqrt_power = compute_sqrt_power_drw(N_GRID, d, 1.0, 50e6)
        xi = jax.random.normal(jax.random.PRNGKey(42), shape=(N_GRID,))

        def f(xi):
            gp = gp_from_xi(xi, sqrt_power, N_GRID)
            return jnp.sum(gp**2)

        grad = jax.grad(f)(xi)

        # Gradient should not be all zeros or all tiny
        grad_rms = float(jnp.sqrt(jnp.mean(grad**2)))
        assert grad_rms > 1e-10, f"GP gradient RMS = {grad_rms}, vanishing"

        # Gradient should not be exploding
        grad_max = float(jnp.max(jnp.abs(grad)))
        assert grad_max < 1e10, f"GP gradient max = {grad_max}, exploding"


class TestGradientThroughPipeline:
    """Test gradient flow through composed pipeline components."""

    def test_full_sfh_pipeline_gradient(self):
        """Gradient flows through PSD -> GP -> mean SFH -> full SFR."""
        log_age_grid = make_log_age_grid(N_GRID)
        d = grid_spacing(log_age_grid)
        age_yr = log_age_to_age_yr(log_age_grid)

        def pipeline(xi, sigma_ps):
            sqrt_power = compute_sqrt_power_drw(N_GRID, float(d), sigma_ps, 50e6)
            gp = gp_from_xi(xi, sqrt_power, N_GRID)
            k0_half = drw_variance(sigma_ps) / 2.0
            sfr_mean = double_powerlaw(age_yr, 1.0, 1.0, 1e9, 10.0)
            sfr = sfr_mean * jnp.exp(gp - k0_half)
            return jnp.sum(jnp.log(sfr + 1e-30))

        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))

        # Gradient w.r.t. xi
        grad_xi = jax.grad(pipeline, argnums=0)(xi, 1.0)
        assert jnp.all(jnp.isfinite(grad_xi)), "Gradient w.r.t. xi not finite"

        # Gradient w.r.t. sigma_ps
        grad_sigma = jax.grad(pipeline, argnums=1)(xi, 1.0)
        assert jnp.isfinite(grad_sigma), "Gradient w.r.t. sigma_ps not finite"

    def test_dust_in_pipeline_gradient(self):
        """Gradient flows through SFH -> dust -> attenuated SED."""
        wave = jnp.linspace(3000, 8000, 50)
        ages = jnp.logspace(6, 10, 30)

        def pipeline(tau_v1, tau_v2, dust_n):
            atten = two_component_dust(
                wave,
                ages,
                tau_v1,
                tau_v2,
                law_bc="power_law",
                law_diff="power_law",
                n_slope=dust_n,
            )
            # Simulate CSP: sum over ages, then sum over wavelengths
            return jnp.sum(atten)

        grads = jax.grad(pipeline, argnums=(0, 1, 2))(0.5, 0.3, -0.7)
        for i, g in enumerate(grads):
            assert jnp.isfinite(g), f"Gradient {i} not finite: {g}"
            assert float(jnp.abs(g)) > 1e-10, f"Gradient {i} vanishing: {g}"
