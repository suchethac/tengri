"""Regression test for attenuation.py float equality safe bug.

Bug: attenuation.py:725-730 — narayanan_z used == for float sentinel detection,
which is JIT-unsafe for traced values.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestAttenuationFloatEqualitySafe:
    """Bug: attenuation.py:725-730 — float == comparison not JIT-safe."""

    def test_narayanan_z_jit_safe(self):
        """narayanan_z should JIT-compile and return correct results."""
        from tengri.components.dust.attenuation import narayanan_z

        wave = jnp.linspace(1000.0, 10000.0, 100)

        @jax.jit
        def _eval(delta, bump):
            return narayanan_z(wave, dust_delta=delta, dust_bump_strength=bump, redshift=0.5)

        # Default sentinel values (delta=-0.2, bump=1.0) should activate redshift scaling
        k_default = _eval(-0.2, 1.0)
        # Non-default values should not activate redshift scaling
        k_custom = _eval(-0.4, 0.5)

        chex.assert_tree_all_finite(k_default)
        chex.assert_tree_all_finite(k_custom)
        # The two should be different (different delta values)
        assert not jnp.allclose(k_default, k_custom)

    def test_narayanan_z_gradient_exists(self):
        """Gradient w.r.t. dust_delta should be finite (not NaN from == comparison)."""
        from tengri.components.dust.attenuation import narayanan_z

        wave = jnp.linspace(1000.0, 10000.0, 50)

        def _sum(delta):
            k = narayanan_z(wave, dust_delta=delta, dust_bump_strength=0.5, redshift=0.3)
            return jnp.sum(k)

        g_jax = float(jax.grad(_sum)(-0.3))
        g_fd = fd_grad(_sum, -0.3)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )
