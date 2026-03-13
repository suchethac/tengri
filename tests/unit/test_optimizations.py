"""Tests for performance optimizations."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from diffsed.utils.optimizations import (
    hartley,
    inverse_hartley,
    gp_from_xi_hartley,
    compute_full_amplitude_drw,
)
from diffsed.models.sfh.gp_sfh import gp_from_xi, compute_sqrt_power_drw
from diffsed.utils.grid import grid_spacing

jax.config.update("jax_enable_x64", True)

N = 256


class TestHartleyTransform:
    """Tests for Hartley transform (NIFTy.re-inspired)."""

    def test_hartley_is_real(self):
        """Hartley transform of real input is real."""
        x = jax.random.normal(jax.random.PRNGKey(0), shape=(N,))
        h = hartley(x)
        assert h.dtype in (jnp.float32, jnp.float64)
        assert not jnp.iscomplexobj(h)

    def test_inverse_roundtrip(self):
        """inverse_hartley(hartley(x)) = x."""
        x = jax.random.normal(jax.random.PRNGKey(1), shape=(N,))
        recovered = inverse_hartley(hartley(x))
        assert_allclose(recovered, x, atol=1e-10)

    def test_hartley_self_inverse(self):
        """hartley(hartley(x)) = N * x (self-reciprocal up to N)."""
        x = jax.random.normal(jax.random.PRNGKey(2), shape=(N,))
        hh = hartley(hartley(x))
        assert_allclose(hh, N * x, atol=1e-8)

    def test_is_jittable(self):
        """Hartley transform can be JIT-compiled."""
        fn = jax.jit(hartley)
        x = jax.random.normal(jax.random.PRNGKey(3), shape=(N,))
        h = fn(x)
        assert h.shape == (N,)

    def test_has_gradients(self):
        """Gradients through Hartley transform are finite."""
        grad_fn = jax.grad(lambda x: jnp.sum(hartley(x) ** 2))
        x = jax.random.normal(jax.random.PRNGKey(4), shape=(N,))
        g = grad_fn(x)
        assert jnp.all(jnp.isfinite(g))


class TestHartleyGP:
    """Test Hartley-based GP generation matches rfft-based version."""

    def test_gp_hartley_has_correct_shape(self):
        """Hartley GP output has correct shape."""
        d = (10.14 - 6.0) / (N - 1)
        amp = compute_full_amplitude_drw(N, d, 1.0, 50e6)
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N,))
        gp = gp_from_xi_hartley(xi, amp)
        assert gp.shape == (N,)

    def test_gp_hartley_zero_xi_gives_zero(self):
        """Zero xi gives zero GP for Hartley version."""
        d = (10.14 - 6.0) / (N - 1)
        amp = compute_full_amplitude_drw(N, d, 1.0, 50e6)
        xi = jnp.zeros(N)
        gp = gp_from_xi_hartley(xi, amp)
        assert_allclose(gp, 0.0, atol=1e-15)

    def test_gp_hartley_has_gradients(self):
        """Gradients through Hartley GP are finite."""
        d = (10.14 - 6.0) / (N - 1)
        amp = compute_full_amplitude_drw(N, d, 1.0, 50e6)
        grad_fn = jax.grad(lambda xi: jnp.sum(gp_from_xi_hartley(xi, amp)))
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N,))
        g = grad_fn(xi)
        assert jnp.all(jnp.isfinite(g))

    def test_hartley_and_rfft_same_statistics(self):
        """Hartley and rfft GP versions produce similar variance."""
        d = (10.14 - 6.0) / (N - 1)

        # rfft version
        sqrt_power = compute_sqrt_power_drw(N, d, 1.0, 50e6)
        # Hartley version
        amp_full = compute_full_amplitude_drw(N, d, 1.0, 50e6)

        key = jax.random.PRNGKey(42)
        n_real = 2000

        # rfft GP batch
        from diffsed.models.sfh.gp_sfh import generate_gp_batch
        batch_rfft = generate_gp_batch(key, sqrt_power, N, n_real)
        var_rfft = float(jnp.var(batch_rfft))

        # Hartley GP batch
        keys = jax.random.split(key, n_real)
        batch_hartley = jax.vmap(
            lambda k: gp_from_xi_hartley(jax.random.normal(k, (N,)), amp_full)
        )(keys)
        var_hartley = float(jnp.var(batch_hartley))

        # Variances should be the same order of magnitude
        ratio = var_hartley / max(var_rfft, 1e-30)
        assert 0.1 < ratio < 10.0, (
            f"Hartley/rfft variance ratio = {ratio:.3f}, expected ~1"
        )
