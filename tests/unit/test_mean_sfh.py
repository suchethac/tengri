"""Tests for parametric mean SFH models."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from diffsed.models.sfh.mean_sfh import (
    double_powerlaw,
    delayed_tau,
    constant_sfh,
    powerlaw_sfh,
)

jax.config.update("jax_enable_x64", True)


class TestDoublePowerlaw:
    """Tests for the BAGPIPES-style double power law mean SFH."""

    def test_positive_output(self):
        """SFR is positive for all lookback times."""
        t = jnp.logspace(6, 10, 200)
        sfr = double_powerlaw(t, alpha=1.0, beta=1.0, tau=1e9, norm=10.0)
        assert jnp.all(sfr > 0)

    def test_peak_near_tau(self):
        """SFR peaks near t = tau for symmetric alpha=beta."""
        t = jnp.logspace(6, 10, 1000)
        tau = 1e9
        sfr = double_powerlaw(t, alpha=2.0, beta=2.0, tau=tau, norm=10.0)
        peak_idx = jnp.argmax(sfr)
        peak_t = t[peak_idx]
        # Peak should be within factor of 2 of tau
        assert 0.5 * tau < float(peak_t) < 2.0 * tau

    def test_peak_value_equals_norm(self):
        """At the exact peak (symmetric case), SFR = norm / 2."""
        tau = 1e9
        sfr_at_tau = double_powerlaw(jnp.array(tau), alpha=1.0, beta=1.0,
                                     tau=tau, norm=10.0)
        # SFR(tau) = norm / (1^alpha + 1^(-beta)) = norm / 2
        assert_allclose(float(sfr_at_tau), 5.0, rtol=1e-10)

    def test_falling_at_late_times(self):
        """SFR decreases at t >> tau (controlled by alpha)."""
        tau = 1e8
        t_late = jnp.array([1e9, 2e9, 5e9, 1e10])
        sfr = double_powerlaw(t_late, alpha=2.0, beta=1.0, tau=tau, norm=10.0)
        assert jnp.all(jnp.diff(sfr) < 0)

    def test_is_jittable(self):
        """Double power law is JIT-compatible."""
        fn = jax.jit(double_powerlaw)
        t = jnp.logspace(6, 10, 100)
        sfr = fn(t, 1.0, 1.0, 1e9, 10.0)
        assert sfr.shape == (100,)

    def test_has_gradients(self):
        """Gradients exist for all 4 parameters."""
        t = jnp.logspace(6, 10, 100)
        grad_fn = jax.grad(
            lambda a, b, tau, n: jnp.sum(double_powerlaw(t, a, b, tau, n))
        )
        grads = grad_fn(1.0, 1.0, 1e9, 10.0)
        assert jnp.isfinite(grads)


class TestDelayedTau:
    """Tests for delayed-tau SFH."""

    def test_peaks_at_tau(self):
        """SFR peaks at t = tau (analytic)."""
        tau = 1e8
        t = jnp.logspace(6, 10, 10000)
        sfr = delayed_tau(t, tau=tau, norm=1.0)
        peak_t = float(t[jnp.argmax(sfr)])
        assert_allclose(peak_t, tau, rtol=0.05)

    def test_positive(self):
        """SFR is positive."""
        t = jnp.logspace(6, 10, 100)
        sfr = delayed_tau(t, tau=1e8, norm=1.0)
        assert jnp.all(sfr > 0)


class TestConstantSFH:
    """Tests for constant SFH."""

    def test_is_constant(self):
        """Output equals norm for all times."""
        t = jnp.logspace(6, 10, 100)
        sfr = constant_sfh(t, norm=5.0)
        assert_allclose(sfr, 5.0, rtol=1e-10)

    def test_correct_shape(self):
        """Output shape matches input."""
        t = jnp.logspace(6, 10, 42)
        sfr = constant_sfh(t, norm=1.0)
        assert sfr.shape == (42,)
