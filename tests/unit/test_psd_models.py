"""Tests for PSD models (T1: PSD integral, analytic properties)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from diffsed.models.sfh.psd_models import (
    psd_drw,
    drw_acf,
    drw_variance,
    psd_to_sqrt_power,
    psd_matern,
)

jax.config.update("jax_enable_x64", True)


class TestPSDDRW:
    """Tests for the damped random walk PSD."""

    def test_psd_drw_shape(self):
        """PSD output shape matches input frequency shape."""
        omega = jnp.linspace(0, 10, 100)
        p = psd_drw(omega, sigma_ps=1.0, tau_ps=1e8)
        assert p.shape == omega.shape

    def test_psd_drw_at_zero_frequency(self):
        """P(0) = sigma_PS^2 * tau_PS."""
        sigma_ps, tau_ps = 2.0, 1e8
        p0 = psd_drw(jnp.array(0.0), sigma_ps, tau_ps)
        assert_allclose(float(p0), sigma_ps**2 * tau_ps, rtol=1e-10)

    def test_psd_drw_positive(self):
        """PSD is strictly positive everywhere."""
        omega = jnp.linspace(0, 100, 1000)
        p = psd_drw(omega, sigma_ps=1.0, tau_ps=1e7)
        assert jnp.all(p > 0)

    def test_psd_drw_monotone_decreasing(self):
        """DRW PSD decreases monotonically with frequency."""
        omega = jnp.linspace(0, 100, 1000)
        p = psd_drw(omega, sigma_ps=1.0, tau_ps=1e7)
        assert jnp.all(jnp.diff(p) <= 0)

    @pytest.mark.parametrize("sigma_ps", [0.5, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("tau_ps", [5e6, 20e6, 50e6, 200e6])
    def test_psd_integral_equals_variance(self, sigma_ps, tau_ps):
        """T1: int P(omega) d_omega / (2*pi) = sigma_PS^2 / 2.

        This is the Wiener-Khinchin theorem: the integral of the PSD
        over all frequencies equals the process variance (zero-lag ACF).
        """
        # Use dense grid for numerical integration
        # Go far past the knee: omega >> 1/tau to capture the tail
        omega_max = 1000.0 / tau_ps
        n_pts = 500_000
        omega = jnp.linspace(0, omega_max, n_pts)
        p = psd_drw(omega, sigma_ps, tau_ps)

        # Numerical integral: int P(omega) d_omega / (2*pi)
        # Factor of 2 for negative frequencies (PSD is symmetric)
        integral = 2.0 * jnp.trapezoid(p, omega) / (2.0 * jnp.pi)

        expected = drw_variance(sigma_ps)
        assert_allclose(float(integral), float(expected), rtol=0.005,
                        err_msg=f"PSD integral failed for sigma={sigma_ps}, tau={tau_ps}")

    def test_psd_drw_is_jittable(self):
        """PSD function can be JIT-compiled."""
        fn = jax.jit(psd_drw)
        omega = jnp.linspace(0, 10, 50)
        p = fn(omega, 1.0, 1e8)
        assert p.shape == (50,)

    def test_psd_drw_has_gradients(self):
        """PSD function has well-defined gradients w.r.t. params."""
        def loss(sigma_ps, tau_ps):
            omega = jnp.linspace(0.01, 10, 50)
            return jnp.sum(psd_drw(omega, sigma_ps, tau_ps))

        grad_fn = jax.grad(loss, argnums=(0, 1))
        g_sigma, g_tau = grad_fn(1.0, 1e8)
        assert jnp.isfinite(g_sigma)
        assert jnp.isfinite(g_tau)


class TestDRWACF:
    """Tests for the DRW analytic autocorrelation function."""

    def test_acf_at_zero_lag(self):
        """ACF(0) = sigma_PS^2 / 2."""
        sigma_ps = 2.0
        acf0 = drw_acf(0.0, sigma_ps, 1e8)
        assert_allclose(float(acf0), drw_variance(sigma_ps), rtol=1e-10)

    def test_acf_positive(self):
        """ACF is positive for all lags."""
        dt = jnp.linspace(0, 1e9, 100)
        acf = drw_acf(dt, 1.0, 1e8)
        assert jnp.all(acf > 0)

    def test_acf_monotone_decreasing(self):
        """ACF decreases with lag."""
        dt = jnp.linspace(0, 1e9, 100)
        acf = drw_acf(dt, 1.0, 1e8)
        assert jnp.all(jnp.diff(acf) <= 0)

    def test_acf_symmetric(self):
        """ACF(dt) = ACF(-dt)."""
        dt = jnp.array([1e7, 5e7, 1e8])
        assert_allclose(
            drw_acf(dt, 1.0, 1e8),
            drw_acf(-dt, 1.0, 1e8),
            rtol=1e-10,
        )


class TestMaternPSD:
    """Tests for Matern PSD (should reduce to DRW for nu=0.5)."""

    def test_matern_nu05_matches_drw(self):
        """Matern with nu=0.5 is equivalent to DRW (up to normalization)."""
        omega = jnp.linspace(0.01, 10, 100)
        sigma_ps = 1.5
        tau_ps = 5e7

        p_drw = psd_drw(omega, sigma_ps, tau_ps)
        # For Matern: variance = sigma_PS^2/2, length_scale = tau_PS
        p_mat = psd_matern(omega, variance=drw_variance(sigma_ps),
                           length_scale=tau_ps, nu=0.5)

        # They should have the same shape (ratio should be constant)
        ratio = p_drw / p_mat
        assert_allclose(float(ratio.std() / ratio.mean()), 0.0, atol=0.05,
                        err_msg="DRW and Matern(nu=0.5) shapes differ")


class TestAmplitudeOperator:
    """Tests for psd_to_sqrt_power."""

    def test_amplitude_positive(self):
        """Amplitude operator is non-negative."""
        p = jnp.array([1.0, 0.5, 0.1, 0.01])
        amp = psd_to_sqrt_power(p, d_grid=0.01)
        assert jnp.all(amp >= 0)

    def test_amplitude_zero_psd(self):
        """Near-zero PSD gives near-zero amplitude (floor at 1e-30)."""
        p = jnp.array([0.0, 0.0, 0.0])
        amp = psd_to_sqrt_power(p, d_grid=0.01)
        assert jnp.all(amp < 1e-10)
