# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-23: wg00_cloudy NaN gradient at tau_k=0.

See ADR / docs/known_bugs.md for full context.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestBug23WG00Gradient:
    """attenuation.py:1198-1202 — grad must be finite at tau_k=0.

    The Weingartner & Draine 2001 attenuation model has a divide-by-zero
    singularity when τ_V = 0. The gradient w.r.t. τ_V is undefined or NaN
    at the boundary.
    """

    @pytest.mark.xfail(reason="BUG-23: NaN gradient trap", strict=True)
    def test_gradient_finite_at_zero_tau(self):
        """jax.grad through wg00_cloudy must not produce NaN at tau_k=0."""
        try:
            from tengri.components.dust.attenuation import wg00_cloudy
        except ImportError:
            pytest.skip("wg00_cloudy not available")

        wave = jnp.array([5500.0])

        def f(tau_v):
            return jnp.sum(wg00_cloudy(wave, dust_tau_v=tau_v))

        grad_jax = float(jax.grad(f)(0.0))
        grad_fd = fd_grad(f, 0.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
