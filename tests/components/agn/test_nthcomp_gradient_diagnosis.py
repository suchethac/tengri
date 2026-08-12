# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostic tests for nthcomp gradient flow issues.

This test suite isolates which operations within nthcomp_lnu_interp produce NaN
gradients when differentiated with respect to gamma parameter.

Problem: `shape * scalar` from nthcomp produces NaN gradient w.r.t. gamma.
Goal: Identify the exact source and propose a fix.
"""

import pytest

pytestmark = pytest.mark.gradient
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# Import the problematic function and its internals
from tengri.components.agn._nthcomp import (
    _clamp_interp_index,
    _get_nthcomp_templates,
    nthcomp_lnu_interp,
)

_GAMMA_JAX, _KTE_JAX, _KTBB_JAX, _NU_JAX, _TABLE_JAX, _TABLE_AVAILABLE = _get_nthcomp_templates()


@pytest.mark.skipif(not _TABLE_AVAILABLE, reason="nthcomp templates not loaded")
class TestNthcompGradientDiagnosis:
    """Isolate which operation in nthcomp causes NaN gradients."""

    def test_clamp_interp_index_gradient(self):
        """FD check: ∂(_clamp_interp_index output sum)/∂val. Only frac has nonzero grad."""
        val = jnp.array(1.5)
        grid = _GAMMA_JAX

        def f(v):
            i_lo, frac = _clamp_interp_index(v, grid)
            return jnp.sum(i_lo) + jnp.sum(frac)

        grad_val = float(jax.grad(f)(val))

        def f_scalar(v):
            return float(f(jnp.array(v)))

        assert_allclose(
            grad_val,
            fd_grad(f_scalar, float(val), eps=0.01),
            rtol=1e-3,
            err_msg="_clamp_interp_index: FD check ∂(sum)/∂val",
        )

    def test_trilinear_interpolation_gradient(self):
        """Test gradient through trilinear interpolation in log space (no resampling)."""
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)

        def f(g):
            ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
            it, ft = _clamp_interp_index(kTe_keV, _KTE_JAX)
            ib, fb = _clamp_interp_index(kTbb_keV, _KTBB_JAX)

            def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
                return _TABLE_JAX[ig + dg, it + dt, ib + db]

            # Trilinear interpolation in log space (same as nthcomp)
            s00 = _c(0, 0, 0) * (1 - fg) + _c(1, 0, 0) * fg
            s10 = _c(0, 1, 0) * (1 - fg) + _c(1, 1, 0) * fg
            s01 = _c(0, 0, 1) * (1 - fg) + _c(1, 0, 1) * fg
            s11 = _c(0, 1, 1) * (1 - fg) + _c(1, 1, 1) * fg
            s0 = s00 * (1 - ft) + s10 * ft
            s1 = s01 * (1 - ft) + s11 * ft
            log_shape = s0 * (1 - fb) + s1 * fb
            return jnp.sum(log_shape)

        grad_val = float(jax.grad(f)(gamma))

        def f_scalar(g):
            return float(f(jnp.array(g)))

        assert_allclose(
            grad_val,
            fd_grad(f_scalar, float(gamma), eps=0.01),
            rtol=1e-3,
            err_msg="trilinear_interp: FD check ∂(∑log_shape)/∂gamma",
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known: jnp.exp(log_shape) overflows → FD=NaN; JAX grad=0. "
            "Fix: clip log_shape before exp."
        ),
    )
    def test_exp_of_interpolated_log_gradient(self):
        """Test gradient through exp(trilinear_log_interp)."""
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)

        def f(g):
            ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
            it, ft = _clamp_interp_index(kTe_keV, _KTE_JAX)
            ib, fb = _clamp_interp_index(kTbb_keV, _KTBB_JAX)

            def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
                return _TABLE_JAX[ig + dg, it + dt, ib + db]

            # Trilinear interpolation
            s00 = _c(0, 0, 0) * (1 - fg) + _c(1, 0, 0) * fg
            s10 = _c(0, 1, 0) * (1 - fg) + _c(1, 1, 0) * fg
            s01 = _c(0, 0, 1) * (1 - fg) + _c(1, 0, 1) * fg
            s11 = _c(0, 1, 1) * (1 - fg) + _c(1, 1, 1) * fg
            s0 = s00 * (1 - ft) + s10 * ft
            s1 = s01 * (1 - ft) + s11 * ft
            log_shape = s0 * (1 - fb) + s1 * fb
            shape = jnp.exp(log_shape)  # THIS IS THE KEY OPERATION
            return jnp.sum(shape)

        grad_val = float(jax.grad(f)(gamma))

        def f_scalar(g):
            return float(f(jnp.array(g)))

        assert_allclose(
            grad_val,
            fd_grad(f_scalar, float(gamma), eps=0.01),
            rtol=1e-3,
            err_msg="exp(log_interp): FD check ∂(∑shape)/∂gamma",
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known: jnp.interp does not propagate gradients through index selection. "
            "JAX grad=0, FD=NaN."
        ),
    )
    def test_jnp_interp_gradient(self):
        """Test gradient through jnp.interp resampling."""
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)

        def f(g):
            # Get interpolated value at table grid
            ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
            it, ft = _clamp_interp_index(kTe_keV, _KTE_JAX)
            ib, fb = _clamp_interp_index(kTbb_keV, _KTBB_JAX)

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
            # Resample to requested nu grid (THIS IS THE KEY OPERATION)
            lnu = jnp.interp(
                nu_test.astype(jnp.float32),
                _NU_JAX,
                shape_on_table_grid,
                left=0.0,
                right=0.0,
            )
            return jnp.sum(lnu)

        grad_val = float(jax.grad(f)(gamma))

        def f_scalar(g):
            return float(f(jnp.array(g)))

        assert_allclose(
            grad_val,
            fd_grad(f_scalar, float(gamma), eps=0.01),
            rtol=1e-3,
            err_msg="jnp.interp resampling: FD check ∂(∑lnu)/∂gamma",
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Known: jnp.interp/exp chain kills gradient. JAX grad=0, FD=NaN.",
    )
    def test_jnp_maximum_clipping_gradient(self):
        """Test gradient through jnp.maximum(lnu, 0.0)."""
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)

        def f(g):
            ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
            it, ft = _clamp_interp_index(kTe_keV, _KTE_JAX)
            ib, fb = _clamp_interp_index(kTbb_keV, _KTBB_JAX)

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
            lnu = jnp.interp(
                nu_test.astype(jnp.float32),
                _NU_JAX,
                shape_on_table_grid,
                left=0.0,
                right=0.0,
            )
            # THIS IS THE KEY OPERATION
            lnu_clipped = jnp.maximum(lnu, 0.0)
            return jnp.sum(lnu_clipped)

        grad_val = float(jax.grad(f)(gamma))

        def f_scalar(g):
            return float(f(jnp.array(g)))

        assert_allclose(
            grad_val,
            fd_grad(f_scalar, float(gamma), eps=0.01),
            rtol=1e-3,
            err_msg="jnp.maximum clipping: FD check ∂(∑lnu_clipped)/∂gamma",
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known: full nthcomp_lnu_interp gradient is zero due to jnp.interp/exp overflow chain."
        ),
    )
    def test_full_nthcomp_lnu_interp_gradient(self):
        """Test gradient through the full nthcomp_lnu_interp function."""
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)

        def f(g):
            shape = nthcomp_lnu_interp(nu_test, g, kTe_keV, kTbb_keV)
            return jnp.sum(shape)

        grad_val = float(jax.grad(f)(gamma))

        def f_scalar(g):
            return float(f(jnp.array(g)))

        assert_allclose(
            grad_val,
            fd_grad(f_scalar, float(gamma), eps=0.01),
            rtol=1e-3,
            err_msg="nthcomp_lnu_interp: FD check ∂(∑shape)/∂gamma",
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known: full nthcomp_lnu_interp gradient is zero due to jnp.interp/exp overflow "
            "chain; scalar multiplication cannot recover lost gradient."
        ),
    )
    def test_nthcomp_multiplied_by_scalar_gradient(self):
        """Test gradient: (nthcomp_shape * scalar) w.r.t. gamma.
        This is the actual failing case from _warm_ring.
        """
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)
        scalar = 1e46

        def f(g):
            shape = nthcomp_lnu_interp(nu_test, g, kTe_keV, kTbb_keV)
            return jnp.sum(shape * scalar)

        grad_val = float(jax.grad(f)(gamma))

        def f_scalar(g):
            return float(f(jnp.array(g)))

        assert_allclose(
            grad_val,
            fd_grad(f_scalar, float(gamma), eps=0.01),
            rtol=1e-3,
            err_msg="nthcomp * scalar: FD check ∂(∑shape * 1e46)/∂gamma",
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known: vmapped nthcomp gradient is zero; jnp.interp/exp overflow kills "
            "gradient flow through nthcomp_lnu_interp."
        ),
    )
    def test_vmap_nthcomp_gradient(self):
        """Test gradient through vmapped nthcomp (like _warm_ring does)."""
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma_array = jnp.array([1.5, 1.6, 1.7])
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)
        scalar = 1e46

        def f(g_arr):
            def per_gamma(g):
                shape = nthcomp_lnu_interp(nu_test, g, kTe_keV, kTbb_keV)
                return jnp.sum(shape * scalar)

            return jnp.sum(jax.vmap(per_gamma)(g_arr))

        grads = jax.grad(f)(gamma_array)
        assert jnp.all(jnp.isfinite(grads)), (
            f"Non-finite gradient through vmapped nthcomp: {grads}"
        )
        # FD check on first component
        g0 = float(gamma_array[0])

        def f0_scalar(g):
            return float(f(gamma_array.at[0].set(g)))

        assert_allclose(
            float(grads[0]),
            fd_grad(f0_scalar, g0, eps=0.01),
            rtol=1e-3,
            err_msg="vmapped nthcomp: FD check ∂(∑shape * 1e46)/∂gamma[0]",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
