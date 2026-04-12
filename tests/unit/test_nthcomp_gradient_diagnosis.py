"""Diagnostic tests for nthcomp gradient flow issues.

This test suite isolates which operations within nthcomp_lnu_interp produce NaN
gradients when differentiated with respect to gamma parameter.

Problem: `shape * scalar` from nthcomp produces NaN gradient w.r.t. gamma.
Goal: Identify the exact source and propose a fix.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

# Import the problematic function and its internals
from tengri.models.agn._nthcomp import (
    _GAMMA_JAX,
    _KTBB_JAX,
    _KTE_JAX,
    _NU_JAX,
    _TABLE_AVAILABLE,
    _TABLE_JAX,
    _clamp_interp_index,
    nthcomp_lnu_interp,
)


@pytest.mark.skipif(not _TABLE_AVAILABLE, reason="nthcomp templates not loaded")
class TestNthcompGradientDiagnosis:
    """Isolate which operation in nthcomp causes NaN gradients."""

    def test_clamp_interp_index_gradient(self):
        """Test gradient through _clamp_interp_index."""
        val = jnp.array(1.5)
        grid = _GAMMA_JAX

        def f(v):
            i_lo, frac = _clamp_interp_index(v, grid)
            return jnp.sum(i_lo) + jnp.sum(frac)

        grad = jax.grad(f)(val)
        assert jnp.isfinite(grad), "Gradient through _clamp_interp_index is NaN"

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

        grad = jax.grad(f)(gamma)
        assert jnp.isfinite(grad), f"Gradient through trilinear interpolation is NaN; grad={grad}"

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

        grad = jax.grad(f)(gamma)
        assert jnp.isfinite(grad), f"Gradient through exp(log_interp) is NaN; grad={grad}"

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

        grad = jax.grad(f)(gamma)
        assert jnp.isfinite(grad), f"Gradient through jnp.interp is NaN; grad={grad}"

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

        grad = jax.grad(f)(gamma)
        assert jnp.isfinite(grad), f"Gradient through jnp.maximum clipping is NaN; grad={grad}"

    def test_full_nthcomp_lnu_interp_gradient(self):
        """Test gradient through the full nthcomp_lnu_interp function."""
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(1.5)
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)

        def f(g):
            shape = nthcomp_lnu_interp(nu_test, g, kTe_keV, kTbb_keV)
            return jnp.sum(shape)

        grad = jax.grad(f)(gamma)
        assert jnp.isfinite(grad), f"Gradient through full nthcomp_lnu_interp is NaN; grad={grad}"

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

        grad = jax.grad(f)(gamma)
        assert jnp.isfinite(grad), f"Gradient through nthcomp * scalar is NaN; grad={grad}"

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

        grad = jax.grad(f)(gamma_array)
        assert jnp.all(jnp.isfinite(grad)), f"Gradient through vmapped nthcomp is NaN; grad={grad}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
