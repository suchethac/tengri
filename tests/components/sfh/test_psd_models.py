# SPDX-License-Identifier: BSD-3-Clause
"""Tests for PSD models (T1: PSD integral, analytic properties)."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sfh.psd_models import (
    drw_acf,
    drw_variance,
    psd_drw,
    psd_matern,
    psd_to_sqrt_power,
)
from tests._jit_parity import assert_jit_matches_eager

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient of scalar f at x."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


pytestmark = pytest.mark.bounds


class TestPSDDRW:
    """Tests for the damped random walk PSD."""

    def test_psd_drw_shape(self):
        """PSD output shape matches input frequency shape."""
        omega = jnp.linspace(0, 10, 100)
        p = psd_drw(omega, psd_sigma=1.0, psd_tau_yr=1e8)
        chex.assert_equal_shape([p, omega])

    def test_psd_drw_at_zero_frequency(self):
        """P(0) = sigma_PS^2 * tau_PS."""
        psd_sigma, psd_tau_yr = 2.0, 1e8
        p0 = psd_drw(jnp.array(0.0), psd_sigma, psd_tau_yr)
        assert_allclose(float(p0), psd_sigma**2 * psd_tau_yr, rtol=1e-10)

    def test_psd_drw_positive(self):
        """PSD is strictly positive everywhere."""
        omega = jnp.linspace(0, 100, 1000)
        p = psd_drw(omega, psd_sigma=1.0, psd_tau_yr=1e7)
        assert jnp.all(p > 0)

    def test_psd_drw_monotone_decreasing(self):
        """DRW PSD decreases monotonically with frequency."""
        omega = jnp.linspace(0, 100, 1000)
        p = psd_drw(omega, psd_sigma=1.0, psd_tau_yr=1e7)
        assert jnp.all(jnp.diff(p) <= 0)

    @pytest.mark.parametrize("psd_sigma", [0.5, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("psd_tau_yr", [5e6, 20e6, 50e6, 200e6])
    def test_psd_integral_equals_variance(self, psd_sigma, psd_tau_yr):
        """T1: int P(omega) d_omega / (2*pi) = sigma_PS^2 / 2.

        This is the Wiener-Khinchin theorem: the integral of the PSD
        over all frequencies equals the process variance (zero-lag ACF).
        """
        # Use dense grid for numerical integration
        # Go far past the knee: omega >> 1/tau to capture the tail
        omega_max = 1000.0 / psd_tau_yr
        n_pts = 500_000
        omega = jnp.linspace(0, omega_max, n_pts)
        p = psd_drw(omega, psd_sigma, psd_tau_yr)

        # Numerical integral: int P(omega) d_omega / (2*pi)
        # Factor of 2 for negative frequencies (PSD is symmetric)
        integral = 2.0 * jnp.trapezoid(p, omega) / (2.0 * jnp.pi)

        expected = drw_variance(psd_sigma)
        assert_allclose(
            float(integral),
            float(expected),
            rtol=0.005,
            err_msg=f"PSD integral failed for sigma={psd_sigma}, tau={psd_tau_yr}",
        )

    def test_psd_drw_is_jittable(self):
        """PSD function can be JIT-compiled."""
        omega = jnp.linspace(0, 10, 50)
        p = assert_jit_matches_eager(psd_drw, omega, 1.0, 1e8)
        chex.assert_shape(p, (50,))

    def test_psd_drw_has_gradients(self):
        """PSD gradients match central FD w.r.t. sigma and tau."""
        omega = jnp.linspace(0.01, 10, 50)

        def loss(psd_sigma, psd_tau_yr):
            return jnp.sum(psd_drw(omega, psd_sigma, psd_tau_yr))

        g_sigma, g_tau = jax.grad(loss, argnums=(0, 1))(1.0, 1e8)

        def f_sigma(s: float) -> float:
            return float(loss(s, 1e8))

        def f_tau(t: float) -> float:
            return float(loss(1.0, t))

        np.testing.assert_allclose(
            float(g_sigma),
            fd_grad(f_sigma, 1.0),
            rtol=1e-3,
            err_msg="psd_drw: FD check ∂(∑PSD)/∂psd_sigma",
        )
        np.testing.assert_allclose(
            float(g_tau),
            fd_grad(f_tau, 1e8, eps=1e4),
            rtol=1e-3,
            err_msg="psd_drw: FD check ∂(∑PSD)/∂psd_tau_yr",
        )


class TestDRWACF:
    """Tests for the DRW analytic autocorrelation function."""

    def test_acf_at_zero_lag(self):
        """ACF(0) = sigma_PS^2 / 2."""
        psd_sigma = 2.0
        acf0 = drw_acf(0.0, psd_sigma, 1e8)
        assert_allclose(float(acf0), drw_variance(psd_sigma), rtol=1e-10)

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
        psd_sigma = 1.5
        psd_tau_yr = 5e7

        p_drw = psd_drw(omega, psd_sigma, psd_tau_yr)
        # For Matern: variance = sigma_PS^2/2, length_scale = tau_PS
        p_mat = psd_matern(
            omega, variance=drw_variance(psd_sigma), length_scale=psd_tau_yr, nu=0.5
        )

        # They should have the same shape (ratio should be constant)
        ratio = p_drw / p_mat
        assert_allclose(
            float(ratio.std() / ratio.mean()),
            0.0,
            atol=0.05,
            err_msg="DRW and Matern(nu=0.5) shapes differ",
        )


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
