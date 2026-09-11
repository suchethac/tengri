# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for wg00_cloudy gradient not zero bug.

Bug: attenuation.py:1198-1202 — safe_tau_k=jnp.where(...,1.0) disconnected the gradient;
gradient of ratio branch was 0 w.r.t. tau_k when tau_k < 1e-10.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestWG00CloudyGradient:
    """Bug: attenuation.py:1198-1202 — gradient disconnected at small tau_v."""

    def test_gradient_finite_near_zero(self):
        """Gradient of wg00_cloudy w.r.t. tau_v must be finite and non-zero near tau_v=0."""
        from tengri.components.dust.attenuation import wg00_cloudy

        wave = jnp.linspace(3000.0, 10000.0, 50)

        def _sum_transmission(tau_v):
            return jnp.sum(wg00_cloudy(wave, tau_v=tau_v))

        # Test near zero — old code had dead gradient here
        g_jax = float(jax.grad(_sum_transmission)(1e-6))
        g_fd = fd_grad(_sum_transmission, 1e-6)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )
        assert g_jax != 0.0, "gradient is zero near tau_v=0 (disconnected)"
        assert np.all(np.isfinite(g_jax)), (
            "`g_jax` is non-finite — non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )

    def test_gradient_finite_at_large_tau(self):
        """Gradient at large tau agrees with FD (wg00_cloudy)."""
        from tengri.components.dust.attenuation import wg00_cloudy

        wave = jnp.linspace(3000.0, 10000.0, 50)

        def _sum_transmission(tau_v):
            return jnp.sum(wg00_cloudy(wave, tau_v=tau_v))

        g_jax = float(jax.grad(_sum_transmission)(2.0))
        np.testing.assert_allclose(
            g_jax,
            fd_grad(lambda t: float(_sum_transmission(t)), 2.0),
            rtol=1e-3,
            err_msg=f"wg00_cloudy large tau: autodiff={g_jax:.4e}",
        )
