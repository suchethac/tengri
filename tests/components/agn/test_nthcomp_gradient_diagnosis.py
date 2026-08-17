# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostic tests for nthcomp gradient flow, one operation at a time.

Each test rebuilds ``nthcomp_lnu_interp`` up to a different operation -- the
trilinear interpolation, the ``exp`` of it, the ``jnp.interp`` resampling, the
``jnp.maximum`` clip, the whole function, the whole function under ``vmap`` --
and compares the analytic gradient w.r.t. gamma against a finite difference.
Structured this way so a broken gradient names the operation that broke it.

Five of these were ``strict=True`` xfails whose reasons said the gradient was
zero and the finite difference NaN. Measured on 2026-08-17: nothing was NaN,
nothing was zero, and every one of them failed for one shared reason unrelated
to the operation it named -- all six probed ``gamma = 1.5``, the table's left
edge. There the analytic derivative is the one-sided slope into the table while
a central difference averages it with the dead clamped side, so analytic is
exactly twice numeric (2.0035 measured at eps=1e-3). Moved to interior gamma
they agree to 0.4% and pass, which is what they now do.

The lesson is in the file's own subject: a finite difference is a measurement,
and at a clamp it is measuring something the analytic gradient is not.
"""

import pytest

pytestmark = pytest.mark.gradient
import jax
import jax.numpy as jnp
import numpy as np
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

    def test_exp_of_interpolated_log_gradient(self):
        """Test gradient through exp(trilinear_log_interp).

        Was a ``strict=True`` xfail reading "jnp.exp(log_shape) overflows ->
        FD=NaN; JAX grad=0". Neither half was true: measured, the gradient is
        7.5e-17 and the finite difference 3.8e-17, both finite. The test failed
        because it probed ``gamma = 1.5``, the table's left edge, where the
        analytic derivative is the one-sided slope into the table and a central
        difference averages it with the dead clamped side -- a factor of exactly
        two (measured 2.0035 at eps=1e-3, converging to 2). At any interior
        gamma the two agree to 0.4%.
        """
        gamma = jnp.array(2.0)
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

    def test_jnp_interp_gradient(self):
        """Test gradient through jnp.interp resampling.

        Was a ``strict=True`` xfail reading "jnp.interp does not propagate
        gradients through index selection. JAX grad=0, FD=NaN." It does
        propagate them: ``jnp.interp`` is differentiable in the *values* it
        interpolates, which is what varies with gamma here. The failure was the
        ``gamma = 1.5`` clamp boundary described above, not the resampling.
        """
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(2.0)
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

    def test_jnp_maximum_clipping_gradient(self):
        """Test gradient through jnp.maximum(lnu, 0.0).

        Was a ``strict=True`` xfail reading "jnp.interp/exp chain kills
        gradient. JAX grad=0, FD=NaN." The chain does not kill it. The clip is
        inactive here -- every sampled ``lnu`` is positive, so ``jnp.maximum``
        passes its gradient straight through -- and the failure was the
        ``gamma = 1.5`` clamp boundary.
        """
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(2.0)
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

    def test_full_nthcomp_lnu_interp_gradient(self):
        """Test gradient through the full nthcomp_lnu_interp function.

        The most misleading of the five: a ``strict=True`` xfail asserting that
        the *shipped* function's gradient "is zero due to jnp.interp/exp
        overflow chain". Measured at this configuration it is 2.82e-16 at
        gamma=2.0 and agrees with a central difference to 0.26%. It was never
        zero and nothing overflowed; the probe sat on the clamp boundary.

        A strict xfail is a claim that something is broken. This one was green
        from 2026-05-21 to 2026-08-17 while the thing it named worked, and a
        reader checking whether nthcomp differentiates would have found it
        asserting, strictly, that it does not.

        ``rtol`` is 1e-2 here and 1e-3 in the three tests above, and the gap is
        the point: those differentiate the table indexing directly, while this
        one goes through the ``custom_jvp``, whose tangent is *itself* a
        one-sided finite difference. Asking an FD-based rule to match another FD
        to 0.1% asks for accuracy it does not have by construction -- measured
        agreement is 0.26% here and 0.37% at gamma=1.7. This test is therefore
        the one that measures the rule's own accuracy, and 1e-2 is the honest
        bound on it, not a tolerance loosened until green.
        """
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(2.0)
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
            rtol=1e-2,
            err_msg="nthcomp_lnu_interp: FD check ∂(∑shape)/∂gamma",
        )

    def test_nthcomp_multiplied_by_scalar_gradient(self):
        """Test gradient: (nthcomp_shape * scalar) w.r.t. gamma.

        This is the actual failing case from _warm_ring. It was a strict xfail
        before, reasoned as "scalar multiplication cannot recover lost gradient" —
        but the loss was not in the ``jnp.interp``/``exp`` chain at all. The old
        reverse rule divided the incoming cotangent by ``max|fd_grad|`` ~1e-17, so
        the ``1e46`` scalar here became ~1e63, overflowed, and a trailing
        ``where(isfinite, ..., 0.0)`` returned zero.

        **This test then passed for the wrong reason, from #1206 to #1822.** The
        kernel returned float32 regardless of the caller, so a 1e46 cotangent
        overflowed float32's 3.4e38 ceiling and *every* quantity here was NaN —
        the autodiff gradient and the finite-difference reference alike.
        ``np.testing.assert_allclose`` defaults to ``equal_nan=True``, so it
        compared NaN to NaN and reported success. #1822 widened the kernel's
        output to the caller's precision, which is the only reason the comparison
        below now actually runs.

        Two guards follow from that. ``isfinite`` is asserted *before* the
        closeness check, so a return to NaN fails loudly instead of silently
        matching. And the probe moved off ``gamma = 1.5``: that is the table's
        exact left edge, where a symmetric difference straddles the clamp — one
        side is flat, so the reference came out at exactly half the true slope
        (measured ratio 0.5000). The boundary itself is covered separately below
        with a one-sided reference.
        """
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma = jnp.array(2.0)  # interior; 1.5 is the clamp boundary
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)
        scalar = 1e46

        def f(g):
            shape = nthcomp_lnu_interp(nu_test, g, kTe_keV, kTbb_keV)
            return jnp.sum(shape * scalar)

        grad_val = float(jax.grad(f)(gamma))

        assert np.isfinite(grad_val), (
            f"gradient is {grad_val} — the kernel is forcing float32 again, so the "
            "1e46 cotangent overflows. Note assert_allclose would ACCEPT this "
            "against a NaN reference (equal_nan=True), which is how it hid (#1822)."
        )

        def f_scalar(g):
            return float(f(jnp.array(g)))

        reference = fd_grad(f_scalar, float(gamma), eps=0.01)
        assert np.isfinite(reference), "the FD reference is NaN and cannot judge anything"

        assert_allclose(
            grad_val,
            reference,
            rtol=5e-3,
            err_msg="nthcomp * scalar: FD check ∂(∑shape * 1e46)/∂gamma",
        )

    def test_gradient_at_the_clamp_boundary_is_one_sided(self):
        """At gamma = 1.5 the derivative is one-sided, and the rule reports it.

        Split out of the test above (#1822). Below 1.5 the table clamps, so the
        function is flat there and a symmetric difference reports half the true
        slope — which is a property of the reference, not of the gradient. The
        one-sided difference is the right comparison at a boundary.
        """
        nu_test = jnp.array([1e14, 2e14, 5e14])
        kTe_keV = jnp.array(0.2)
        kTbb_keV = jnp.array(0.001)
        scalar = 1e46

        def f(g):
            return jnp.sum(nthcomp_lnu_interp(nu_test, g, kTe_keV, kTbb_keV) * scalar)

        def fs(g):
            return float(f(jnp.array(g)))

        edge = 1.5
        grad_val = float(jax.grad(f)(jnp.array(edge)))
        assert np.isfinite(grad_val), f"gradient at the boundary is {grad_val}"

        eps = 0.01
        one_sided = (fs(edge + eps) - fs(edge)) / eps
        assert_allclose(grad_val, one_sided, rtol=0.05)

        # And the clamp itself: the function really is flat below the edge, which
        # is what makes a symmetric reference wrong here rather than merely noisy.
        assert fs(edge - eps) == fs(edge), (
            "gamma below the table's left edge is no longer clamped; the one-sided "
            "reasoning above needs revisiting"
        )

    def test_vmap_nthcomp_gradient(self):
        """Test gradient through vmapped nthcomp (like _warm_ring does).

        Was a ``strict=True`` xfail reading "vmapped nthcomp gradient is zero".
        It is not, and vmap was never the issue: the FD check ran on
        ``gamma_array[0] = 1.5``, the clamp boundary, so this inherited the same
        factor of two as its four siblings. Every element now sits inside the
        table, which is also what ``_warm_ring`` samples.

        ``rtol`` is 1e-2 for the same reason as the test above: this goes
        through the ``custom_jvp``, whose tangent is a finite difference itself.
        """
        nu_test = jnp.array([1e14, 2e14, 5e14])
        gamma_array = jnp.array([1.7, 2.0, 2.3])
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
            rtol=1e-2,
            err_msg="vmapped nthcomp: FD check ∂(∑shape * 1e46)/∂gamma[0]",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
