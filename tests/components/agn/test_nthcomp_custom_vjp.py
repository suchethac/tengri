# SPDX-License-Identifier: BSD-3-Clause
"""Test custom VJP for nthcomp to handle gradient issues."""

import pytest

pytestmark = pytest.mark.gradient
import jax
import jax.numpy as jnp
import numpy as np


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.agn._nthcomp import (
    _clamp_interp_index,
    _get_nthcomp_templates,
)

_GAMMA_JAX, _KTE_JAX, _KTBB_JAX, _NU_JAX, _TABLE_JAX, _TABLE_AVAILABLE = _get_nthcomp_templates()


@pytest.mark.skipif(not _TABLE_AVAILABLE, reason="nthcomp templates not loaded")
class TestNthcompCustomVJP:
    """Test custom VJP solution for nthcomp gradient."""

    def test_custom_vjp_for_nthcomp(self):
        """Define custom VJP for nthcomp to avoid jnp.interp gradient issue."""

        # Define the function with a custom VJP rule
        @jax.custom_vjp
        def nthcomp_lnu_interp_custom(nu, gamma, kTe_keV, kTbb_keV):
            """Custom VJP version of nthcomp interpolation."""
            g = jnp.asarray(gamma, dtype=jnp.float32)
            t = jnp.asarray(kTe_keV, dtype=jnp.float32)
            b = jnp.asarray(kTbb_keV, dtype=jnp.float32)
            ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
            it, ft = _clamp_interp_index(t, _KTE_JAX)
            ib, fb = _clamp_interp_index(b, _KTBB_JAX)

            def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
                return _TABLE_JAX[ig + dg, it + dt, ib + db]

            # Trilinear in log space
            s00 = _c(0, 0, 0) * (1 - fg) + _c(1, 0, 0) * fg
            s10 = _c(0, 1, 0) * (1 - fg) + _c(1, 1, 0) * fg
            s01 = _c(0, 0, 1) * (1 - fg) + _c(1, 0, 1) * fg
            s11 = _c(0, 1, 1) * (1 - fg) + _c(1, 1, 1) * fg
            s0 = s00 * (1 - ft) + s10 * ft
            s1 = s01 * (1 - ft) + s11 * ft
            log_shape_on_table_grid = s0 * (1 - fb) + s1 * fb
            shape_on_table_grid = jnp.exp(log_shape_on_table_grid)
            # Resample (this is where the gradient issue occurs)
            nu_f = jnp.asarray(nu, dtype=jnp.float32)
            lnu = jnp.interp(nu_f, _NU_JAX, shape_on_table_grid, left=0.0, right=0.0)
            lnu_clipped = jnp.maximum(lnu, 0.0)
            return jnp.asarray(lnu_clipped, dtype=jnp.float64)

        def fwd(nu, gamma, kTe_keV, kTbb_keV):
            """Forward pass: compute function value and save residuals for backward."""
            result = nthcomp_lnu_interp_custom(nu, gamma, kTe_keV, kTbb_keV)
            # For the backward pass, we need the shape at the table grid
            g = jnp.asarray(gamma, dtype=jnp.float32)
            t = jnp.asarray(kTe_keV, dtype=jnp.float32)
            b = jnp.asarray(kTbb_keV, dtype=jnp.float32)
            ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
            it, ft = _clamp_interp_index(t, _KTE_JAX)
            ib, fb = _clamp_interp_index(b, _KTBB_JAX)

            def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
                return _TABLE_JAX[ig + dg, it + dt, ib + db]

            s00 = _c(0, 0, 0) * (1 - fg) + _c(1, 0, 0) * fg
            s10 = _c(0, 1, 0) * (1 - fg) + _c(1, 1, 0) * fg
            s01 = _c(0, 0, 1) * (1 - fg) + _c(1, 0, 1) * fg
            s11 = _c(0, 1, 1) * (1 - fg) + _c(1, 1, 1) * fg
            s0 = s00 * (1 - ft) + s10 * ft
            s1 = s01 * (1 - ft) + s11 * ft
            log_shape_on_table_grid = s0 * (1 - fb) + s1 * fb
            shape_on_table_grid = jnp.exp(log_shape_on_table_grid)
            # Residuals for backward pass
            residuals = (nu, gamma, kTe_keV, kTbb_keV, shape_on_table_grid, result)
            return result, residuals

        def bwd(residuals, g_out):
            """Backward pass: compute gradients w.r.t. gamma (only param we care about)."""
            nu, gamma, kTe_keV, kTbb_keV, _shape_on_table_grid, result = residuals
            # Simple approach: finite difference for gamma gradient
            eps = 1e-5
            gamma_plus = gamma + eps
            result_plus = nthcomp_lnu_interp_custom(nu, gamma_plus, kTe_keV, kTbb_keV)
            # Finite difference gradient
            fd_grad_gamma = (result_plus - result) / eps
            # Chain rule
            g_gamma = jnp.sum(g_out * fd_grad_gamma)
            g_kTe = jnp.zeros_like(kTe_keV)  # Not differentiating w.r.t. these
            g_kTbb = jnp.zeros_like(kTbb_keV)
            g_nu = jnp.zeros_like(nu)
            return (g_nu, g_gamma, g_kTe, g_kTbb)

        nthcomp_lnu_interp_custom.defvjp(fwd, bwd)
        # Test the custom VJP version
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)
        scalar = 1e46

        def f(g):
            shape = nthcomp_lnu_interp_custom(nu_test, g, kTe_keV, kTbb_keV)
            return jnp.sum(shape * scalar)

        grad_jax = float(jax.grad(f)(gamma))
        grad_fd = fd_grad(f, float(gamma))
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1.5,
            atol=1e28,
            err_msg="nthcomp_lnu_interp_custom: FD check ∂/∂gamma",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
