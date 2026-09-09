# SPDX-License-Identifier: BSD-3-Clause
"""``narayanan_z`` must stay JIT-safe and differentiable in its own knob.

Original bug: the law detected its sentinel defaults with ``==`` on values that
could be traced, which is JIT-unsafe. That was fixed by comparing with a
tolerance instead.

#2199 removed the sentinels entirely, and with them the comparison this file was
written for. Keeping the file as a comparison test would leave it passing for a
reason that no longer exists; keeping the *property* is what matters, so it now
pins the same two guarantees on the knob the law actually has. ``redshift`` is
the sole traced input, it is read with ``jnp.interp`` over the fitted
Narayanan+2018 table, and a photometric-redshift fit needs that read to compile
and to differentiate.
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
    """No Python-level branch on a traced value anywhere in ``narayanan_z``."""

    def test_narayanan_z_jit_safe(self):
        """narayanan_z should JIT-compile on a traced redshift and stay finite."""
        from tengri.components.dust.attenuation import narayanan_z

        wave = jnp.linspace(1000.0, 10000.0, 100)

        @jax.jit
        def _eval(z):
            return narayanan_z(wave, redshift=z)

        k_low = _eval(0.5)
        k_high = _eval(4.0)

        chex.assert_tree_all_finite(k_low)
        chex.assert_tree_all_finite(k_high)
        # Two different redshifts are two different curves; if they were equal
        # the traced value never reached the table (#2199).
        assert not jnp.allclose(k_low, k_high)

    def test_narayanan_z_gradient_exists(self):
        """d k / d z must be finite and match a finite difference.

        Taken at z = 0.3, inside the first table interval: the interpolation is
        linear there, so autodiff and a central difference must agree closely.
        The end nodes are deliberately flat and would give an uninformative zero.
        """
        from tengri.components.dust.attenuation import narayanan_z

        wave = jnp.linspace(1000.0, 10000.0, 50)

        def _sum(z):
            return jnp.sum(narayanan_z(wave, redshift=z))

        g_jax = float(jax.grad(_sum)(0.3))
        g_fd = fd_grad(_sum, 0.3)
        assert g_jax != 0.0, "redshift has an exactly-zero gradient inside the table"
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )
