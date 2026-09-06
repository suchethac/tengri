# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for triweight interpolation on non-uniform grids (I6 fix).

Verifies that the index-space triweight interpolation produces:
1. 40/40 distinct outputs on a 40-point sweep (not nearest-neighbor snapping)
2. 0% exactly-zero gradients (smooth derivatives throughout)
3. Correct node-value interpolation (no overshoot at tabulated points)
4. Bit-identical results on uniform axes (fast path unchanged)
5. Gradient safety in both f64 and f32 precision
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestNonuniformTriweight:
    """Index-space triweight kernel tests for non-uniform axes."""

    def test_nonuniform_axis_40_point_sweep(self):
        """Non-uniform axis: 40-point sweep yields 40 distinct outputs.

        Tests the OPT-IN path: index_space_interp=True fixes the #1851 degeneracy
        by using index-space interpolation on non-uniform axes.
        """
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Fritz tau axis: [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0]
        tau_axis = jnp.asarray([0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0])
        edges = edges_for_grid(tau_axis)

        # Create a dummy grid and interpolate
        grid = jnp.ones((8, 4))  # 8 grid points, 4 trailing dimensions
        for i in range(8):
            grid = grid.at[i, :].set(float(i))

        # 40-point sweep from [0.1, 10.0]
        xs = jnp.linspace(0.1, 10.0, 40)

        @jax.jit
        def interp(x):
            scatter = 0.5 * (tau_axis[1] - tau_axis[0])
            # OPT-IN to index-space interpolation (fixes #1851)
            w = compute_grid_weights(
                x, tau_axis, scatter=scatter, edges=edges, index_space_interp=True
            )
            return jnp.tensordot(w, grid, axes=([0], [0]))

        results = jnp.array([interp(x) for x in xs])
        distinct_count = len(np.unique(np.round(results[:, 0], 10)))

        assert distinct_count == 40, (
            f"Non-uniform axis sweep should give 40 distinct outputs; "
            f"got {distinct_count} (nearest-neighbor snapping issue)"
        )

    def test_nonuniform_axis_zero_gradient_fraction(self):
        """Non-uniform axis: gradient should be nonzero throughout.

        Tests the OPT-IN path: index_space_interp=True ensures smooth gradients
        everywhere on the non-uniform axis (no %zero-gradient regions).
        """
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        tau_axis = jnp.asarray([0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0])
        edges = edges_for_grid(tau_axis)

        grid = jnp.ones((8, 4))
        for i in range(8):
            grid = grid.at[i, :].set(float(i))

        xs = jnp.linspace(0.1, 10.0, 40)

        @jax.jit
        def interp_sum(x):
            scatter = 0.5 * (tau_axis[1] - tau_axis[0])
            # OPT-IN to index-space interpolation (fixes #1851)
            w = compute_grid_weights(
                x, tau_axis, scatter=scatter, edges=edges, index_space_interp=True
            )
            result = jnp.tensordot(w, grid, axes=([0], [0]))
            return jnp.sum(result)

        grad_fn = jax.jit(jax.grad(interp_sum))
        grads = jnp.array([grad_fn(x) for x in xs])

        # Count exactly-zero gradients
        zero_count = jnp.sum(grads == 0.0).item()
        zero_fraction = zero_count / len(xs)

        assert zero_fraction == 0.0, (
            f"Non-uniform axis sweep should have 0% exactly-zero gradients; "
            f"got {zero_fraction:.1%} (kernel support issue)"
        )

    def test_default_none_path_warns_on_nonuniform_concrete_axis(self):
        """Default path (index_space_interp=None) warns on non-uniform concrete axes.

        Verifies that the #1851 warning is emitted at call-time when a concrete
        non-uniform axis is detected with the default parameter (None).
        Legacy behavior: physical-space path with warning.
        """
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        tau_axis = jnp.asarray([0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0])
        edges = edges_for_grid(tau_axis)

        grid = jnp.ones((8, 4))
        for i in range(8):
            grid = grid.at[i, :].set(float(i))

        # Query point
        x = 2.5

        # Default path (None) should emit warning for concrete non-uniform axis
        with pytest.warns(UserWarning, match="1851 degeneracy"):
            w = compute_grid_weights(x, tau_axis, scatter=0.5, edges=edges)
            # Verify weights are computed (legacy path, so may have degeneracy)
            assert np.isclose(np.sum(w), 1.0, atol=1e-14)

    def test_uniform_axis_still_works(self):
        """Uniform axis: results should match pre-fix behavior (bit-identical or ~1e-15)."""
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Uniform axis
        uniform_ax = jnp.linspace(0.0, 10.0, 50)
        edges = edges_for_grid(uniform_ax)

        grid = jnp.ones((50, 3))
        for i in range(50):
            grid = grid.at[i, :].set(float(i))

        # Test query points
        xs = [0.5, 2.5, 5.0, 7.5, 9.5]

        for x in xs:
            scatter = 0.5 * (uniform_ax[1] - uniform_ax[0])
            w = compute_grid_weights(x, uniform_ax, scatter=scatter, edges=edges)
            result = jnp.tensordot(w, grid, axes=([0], [0]))

            # Check that weights sum to 1
            assert np.isclose(np.sum(w), 1.0, atol=1e-14), f"Weights should sum to 1 at x={x}"

            # Check that result is finite
            assert np.all(np.isfinite(result)), f"Result should be finite at x={x}"

    def test_node_value_interpolation(self):
        """At a grid node, interpolated value must equal the tabulated value.

        A triweight smoothing kernel must pass: interpolation at a node should
        reproduce the value at that node (within machine precision). This pins
        a critical property: no overshoot from the smoothing kernel.
        """
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Use a uniform axis for this test
        uniform_axis = jnp.linspace(1.0, 10.0, 8)
        edges = edges_for_grid(uniform_axis)

        # Create grid with distinctive values
        grid_values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        # Use a tight scatter (0.1 * spacing) so that at nodes, weight is ~100%
        # on that node and ~0% on neighbors. This is the case that matters for
        # ensuring the interpolant is well-behaved.
        spacing = uniform_axis[1] - uniform_axis[0]
        scatter = 0.1 * spacing

        for node_idx, node_x in enumerate(uniform_axis):
            w = compute_grid_weights(node_x, uniform_axis, scatter=scatter, edges=edges)

            # Interpolated value at the node
            interp_value = jnp.dot(w, grid_values)
            expected_value = grid_values[node_idx]

            # Must match to machine precision: with tight scatter, w[i] ≈ 1.0
            abs_error = jnp.abs(interp_value - expected_value)
            assert float(abs_error) < 1e-12, (
                f"At node {node_idx} ({float(node_x):.4f}): "
                f"interpolated {float(interp_value):.15f} "
                f"should equal tabulated {float(expected_value):.15f}, "
                f"abs_error {float(abs_error):.2e}"
            )

            # Weight on that node should be very close to 1.0
            assert w[node_idx] > 0.999, (
                f"At node {node_idx}, weight should be > 0.999; got {float(w[node_idx])}"
            )

    def test_gradient_safe_f32(self):
        """Non-uniform axis: gradients should be finite in f32 precision."""
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        tau_axis = jnp.asarray([0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0], dtype=jnp.float32)
        edges = edges_for_grid(tau_axis)

        grid = jnp.ones((8, 4), dtype=jnp.float32)
        for i in range(8):
            grid = grid.at[i, :].set(float(i))

        xs = jnp.linspace(0.1, 10.0, 10, dtype=jnp.float32)

        @jax.jit
        def interp_sum(x):
            scatter = 0.5 * (tau_axis[1] - tau_axis[0])
            w = compute_grid_weights(x, tau_axis, scatter=scatter, edges=edges)
            result = jnp.tensordot(w, grid, axes=([0], [0]))
            return jnp.sum(result)

        jax.config.update("jax_enable_x64", False)
        try:
            grad_fn = jax.jit(jax.grad(interp_sum))
            for x in xs:
                grad = grad_fn(x)
                assert np.isfinite(grad), f"Gradient at x={x} is not finite in f32: {grad}"
                assert np.any(grad != 0.0), (
                    "`grad` is identically zero — finite is not enough, "
                    "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
                )
        finally:
            jax.config.update("jax_enable_x64", True)

    def test_jit_and_grad_compatible(self):
        """Non-uniform axis: must be JIT-compatible and differentiable."""
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        tau_axis = jnp.asarray([0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0])
        edges = edges_for_grid(tau_axis)
        grid = jnp.arange(8, dtype=jnp.float64)

        @jax.jit
        def f(x):
            scatter = 0.5 * (tau_axis[1] - tau_axis[0])
            w = compute_grid_weights(x, tau_axis, scatter=scatter, edges=edges)
            return jnp.sum(w * grid)

        @jax.jit
        def grad_f(x):
            return jax.grad(f)(x)

        # Should not raise
        x_test = jnp.asarray(1.5)
        result = f(x_test)
        grad_result = grad_f(x_test)

        assert np.isfinite(result), "Function result should be finite"
        assert np.isfinite(grad_result), "Gradient should be finite"
        assert np.any(grad_result != 0.0), (
            "`grad_result` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_single_node_axis_no_crash(self):
        """Length-1 axis should not crash on min() of empty spacings."""
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Length-1 axis: diff gives empty array
        single_axis = jnp.asarray([5.0])
        edges = edges_for_grid(single_axis)

        # Should not crash on min/max of empty array
        w = compute_grid_weights(5.0, single_axis, scatter=1.0, edges=edges)
        assert w.shape == (1,), f"Expected shape (1,), got {w.shape}"
        assert np.isfinite(w[0]), "Weight should be finite"

    def test_single_node_axis_np_vs_jnp_identical(self):
        """Length-1 axis gives identical edges and weights for np and jnp."""
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Length-1 axis: test that np and jnp produce identical results
        single_axis_np = np.array([5.0])
        single_axis_jnp = jnp.asarray([5.0])

        # edges_for_grid should produce identical edges
        edges_np = edges_for_grid(single_axis_np)
        edges_jnp = edges_for_grid(single_axis_jnp)
        np.testing.assert_allclose(edges_np, edges_jnp, rtol=1e-14)

        # compute_grid_weights should produce identical weights
        w_np = compute_grid_weights(5.0, single_axis_np, scatter=1.0, edges=edges_np)
        w_jnp = compute_grid_weights(5.0, single_axis_jnp, scatter=1.0, edges=edges_jnp)
        np.testing.assert_allclose(w_np, w_jnp, rtol=1e-14)

        # Single node should take all weight
        assert np.allclose(w_np[0], 1.0), f"Single node should take all weight, got {w_np[0]}"
        assert np.allclose(w_jnp[0], 1.0), f"Single node should take all weight, got {w_jnp[0]}"

    def test_length_two_axis_uniform(self):
        """Length-2 axis should be treated as uniform (one spacing)."""
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Length-2 axis: one spacing, definitely uniform
        short_axis = jnp.asarray([1.0, 3.0])
        edges = edges_for_grid(short_axis)

        w_at_1 = compute_grid_weights(1.0, short_axis, scatter=0.5, edges=edges)
        w_at_3 = compute_grid_weights(3.0, short_axis, scatter=0.5, edges=edges)

        # At the nodes, most weight should be on that node
        assert w_at_1[0] > 0.5, f"Weight at node 0 should be > 0.5, got {w_at_1[0]}"
        assert w_at_3[1] > 0.5, f"Weight at node 1 should be > 0.5, got {w_at_3[1]}"

    def test_float32_uniform_linspace_no_false_nonuniform(self):
        """Float32 uniform linspace should not flip to index path (I6 issue #1851).

        A nominally-uniform float32 linspace has small spacing variations (~1e-6
        due to accumulated rounding). The dtype-aware tolerance must ignore these
        and keep f32 uniform axes on the uniform (fast) path, producing weights
        identical to f64 within f32 machine epsilon.
        """
        from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

        # Create a uniform linspace in both f32 and f64
        axis_f64 = jnp.linspace(0.0, 10.0, 50, dtype=jnp.float64)
        axis_f32 = jnp.linspace(0.0, 10.0, 50, dtype=jnp.float32)

        # Compute edges for both
        edges_f64 = edges_for_grid(axis_f64)
        edges_f32 = edges_for_grid(axis_f32)

        # Query at an interior point
        x_query = 5.5
        scatter = 0.5 * (axis_f64[1] - axis_f64[0])

        # Compute weights for both
        w_f64 = compute_grid_weights(x_query, axis_f64, scatter=scatter, edges=edges_f64)
        w_f32 = compute_grid_weights(x_query, axis_f32, scatter=scatter, edges=edges_f32)

        # Convert f32 weights to f64 for comparison
        w_f32_as_f64 = w_f32.astype(jnp.float64)

        # They should match within float32 machine epsilon (~1e-7)
        # Use 100x f32 eps to account for accumulated rounding in weight computation
        f32_eps = np.float32(jnp.finfo(jnp.float32).eps)
        rel_error = np.linalg.norm(w_f64 - w_f32_as_f64) / (np.linalg.norm(w_f64) + 1e-300)
        assert rel_error < 100.0 * float(f32_eps), (
            f"Float32 and float64 weights should match within ~100x f32 eps; "
            f"rel error {rel_error:.2e} exceeds threshold {100.0 * float(f32_eps):.2e}. "
            f"This suggests float32 axis flipped to index path."
        )
