# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for grid interpolation kernel axis requirements.

The triweight grid kernel (edges_for_grid and compute_grid_weights) requires
concrete (non-traced) axes that:
- Have at least 2 nodes (n >= 2)
- Are strictly ascending (d[0] > 0 and all spacings strictly positive)

This suite guards both requirements:
- Size-1 axes raise ValueError from edges_for_grid
- Descending axes raise ValueError from compute_grid_weights
- Traced axes (under jax.jit, jax.eval_shape, etc.) do NOT raise
- Ascending axes return expected weights
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

pytestmark = pytest.mark.unit


class TestEdgesForGridContract:
    """Guard edges_for_grid contract: concrete axes must be ascending with n >= 2."""

    def test_ascending_axis_returns_edges(self):
        """Ascending axis returns n+1 edges."""
        grid = jnp.array([0.0, 1.0, 2.0])
        edges = edges_for_grid(grid)
        assert edges.shape == (4,), f"Expected 4 edges, got {edges.shape}"
        # For a uniform grid [0, 1, 2], edges are [-0.5, 0.5, 1.5, 2.5]
        np.testing.assert_allclose(edges, jnp.array([-0.5, 0.5, 1.5, 2.5]))

    def test_nonuniform_ascending_axis(self):
        """Non-uniform ascending axis returns edges."""
        grid = jnp.array([0.0, 0.5, 3.0])
        edges = edges_for_grid(grid)
        assert edges.shape == (4,)
        # edges: grid[0] - (grid[1]-grid[0])/2 = 0 - 0.25 = -0.25
        # interior: (grid[0]+grid[1])/2 = 0.25, (grid[1]+grid[2])/2 = 1.75
        # grid[-1] + (grid[-1]-grid[-2])/2 = 3 + 1.25 = 4.25
        np.testing.assert_allclose(edges, jnp.array([-0.25, 0.25, 1.75, 4.25]))

    def test_size_0_axis_raises(self):
        """Empty concrete axis raises ValueError."""
        grid_np = np.array([])
        with pytest.raises(ValueError, match="empty"):
            edges_for_grid(grid_np)

    def test_size_1_axis_np_vs_jnp(self):
        """Size-1 axis produces identical edges for np and jnp."""
        grid_np = np.array([5.0])
        grid_jnp = jnp.asarray([5.0])

        edges_np = edges_for_grid(grid_np)
        edges_jnp = edges_for_grid(grid_jnp)

        np.testing.assert_allclose(edges_np, edges_jnp, rtol=1e-14)
        # Edges for size-1 should be [value, value]
        np.testing.assert_allclose(edges_np, jnp.array([5.0, 5.0]))

    def test_descending_axis_raises(self):
        """Descending concrete axis raises ValueError."""
        grid_np = np.array([3.0, 2.0, 1.0])
        with pytest.raises(ValueError, match="not strictly ascending"):
            edges_for_grid(grid_np)

    def test_descending_axis_message_contains_values(self):
        """Descending axis error message includes first and last values."""
        grid_np = np.array([5.0, 3.0, 1.0])
        with pytest.raises(ValueError) as exc_info:
            edges_for_grid(grid_np)
        msg = str(exc_info.value)
        assert "first=5" in msg or "first = 5" in msg.lower()
        assert "last=1" in msg or "last = 1" in msg.lower()

    def test_descending_axis_message_reverse_guidance(self):
        """Descending axis message advises to reverse together."""
        grid_np = np.array([10.0, 5.0, 1.0])
        with pytest.raises(ValueError) as exc_info:
            edges_for_grid(grid_np)
        msg = str(exc_info.value)
        assert "reverse" in msg.lower()
        assert "together" in msg.lower()

    def test_flat_axis_raises(self):
        """Flat (equal-node) axis raises ValueError."""
        grid_np = np.array([2.0, 2.0, 2.0])
        with pytest.raises(ValueError, match="not strictly ascending"):
            edges_for_grid(grid_np)

    def test_descending_integer_axis_raises(self):
        """Descending integer axis is rejected (validates all numeric dtypes)."""
        grid_int = np.array([10, 5, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="not strictly ascending"):
            edges_for_grid(grid_int)

    def test_traced_axis_does_not_raise(self):
        """Traced axis (under jit/eval_shape) does not raise."""

        def call_edges(grid):
            return edges_for_grid(grid)

        # eval_shape traces without executing, so it does not raise
        grid_shape = jax.eval_shape(call_edges, jnp.zeros(3))
        assert grid_shape.shape == (4,), "jit-traced call should return shape (4,)"


class TestComputeGridWeightsContract:
    """Guard compute_grid_weights contract: n >= 2 and strictly ascending."""

    def test_uniform_ascending_axis_weights_sum_to_one(self):
        """Uniform ascending axis returns weights summing to 1."""
        grid = jnp.linspace(0.0, 10.0, 5)
        x = 3.5
        w = compute_grid_weights(x, grid)
        assert w.shape == (5,), f"Expected 5 weights, got {w.shape}"
        np.testing.assert_allclose(jnp.sum(w), 1.0, rtol=1e-10)

    def test_nonuniform_ascending_weights_sum_to_one(self):
        """Non-uniform ascending axis returns weights summing to 1."""
        grid = jnp.array([0.0, 0.5, 3.0])
        x = 1.5
        w = compute_grid_weights(x, grid, index_space_interp=False)
        np.testing.assert_allclose(jnp.sum(w), 1.0, rtol=1e-10)

    def test_size_0_axis_raises(self):
        """Empty concrete axis raises ValueError."""
        grid_np = np.array([])
        with pytest.raises(ValueError, match="empty"):
            compute_grid_weights(2.5, grid_np)

    def test_size_1_axis_all_weight(self):
        """Size-1 axis returns weight of 1.0 for the single node."""
        grid_np = np.array([5.0])
        grid_jnp = jnp.asarray([5.0])

        w_np = compute_grid_weights(5.0, grid_np)
        w_jnp = compute_grid_weights(5.0, grid_jnp)

        # Both should give all weight to the single node
        np.testing.assert_allclose(w_np, jnp.array([1.0]))
        np.testing.assert_allclose(w_jnp, jnp.array([1.0]))
        np.testing.assert_allclose(w_np, w_jnp, rtol=1e-14)

    def test_descending_axis_raises(self):
        """Descending concrete axis raises ValueError."""
        grid_np = np.array([5.0, 3.0, 1.0])
        with pytest.raises(ValueError, match="not strictly ascending"):
            compute_grid_weights(2.5, grid_np)

    def test_flat_axis_raises(self):
        """Flat axis raises ValueError."""
        grid_np = np.array([2.0, 2.0, 2.0])
        with pytest.raises(ValueError, match="not strictly ascending"):
            compute_grid_weights(2.0, grid_np)

    def test_traced_axis_does_not_raise(self):
        """Traced axis (under jit) does not raise even if size-1."""

        def call_weights(grid):
            return compute_grid_weights(1.0, grid)

        # jit traces without executing size/shape checks, so a size-1 grid
        # must not raise. This ran inside a ``try`` that skipped on
        # ``(ValueError, IndexError)`` under the message "platform-dependent" --
        # in a test named ``_does_not_raise``, which made the raise it exists to
        # forbid report as a skip. CI pins JAX_PLATFORMS=cpu; a raise here is a
        # failure.
        result = jax.jit(call_weights)(jnp.array([2.0]))
        assert result.shape == (1,)

    def test_inside_jit_no_raise_on_traced_grid(self):
        """Traced grid inside jit does not raise, only at call time."""

        def use_weights(grid):
            # Call compute_grid_weights inside jit
            return compute_grid_weights(1.5, grid)

        # jit should work with non-degenerate grids
        grid = jnp.linspace(0.0, 3.0, 4)
        result = jax.jit(use_weights)(grid)
        assert result.shape == (4,)

    def test_eval_shape_does_not_raise_on_concrete(self):
        """eval_shape on size-1 grid does not raise (grid is traced)."""
        grid = jnp.array([1.0])
        # eval_shape traces the grid, so no concrete check occurs
        shape = jax.eval_shape(lambda g: compute_grid_weights(0.5, g), grid)
        assert shape.shape == (1,)


class TestEdgesForGridAgainstInterpetation:
    """Verify edges_for_grid produces geometrically sensible output."""

    def test_edges_bracket_nodes(self):
        """Edges should bracket nodes (all nodes between first and last edge)."""
        grid = jnp.array([1.0, 2.0, 4.0])
        edges = edges_for_grid(grid)
        # Edges: [lo, e01, e12, hi]
        # lo < grid[0], e01 in (grid[0], grid[1]), e12 in (grid[1], grid[2]), hi > grid[2]
        assert edges[0] < grid[0]
        assert grid[0] < edges[1] < grid[1]
        assert grid[1] < edges[2] < grid[2]
        assert edges[3] > grid[2]

    def test_edges_symmetric_on_uniform(self):
        """On uniform grid, edges should be symmetric around nodes."""
        grid = jnp.linspace(0.0, 3.0, 4)  # 0, 1, 2, 3
        edges = edges_for_grid(grid)
        # For uniform grid with spacing 1:
        # edges: [-0.5, 0.5, 1.5, 2.5, 3.5]
        spacing = float(grid[1] - grid[0])
        expected = jnp.array(
            [
                grid[0] - spacing / 2.0,
                (grid[0] + grid[1]) / 2.0,
                (grid[1] + grid[2]) / 2.0,
                (grid[2] + grid[3]) / 2.0,
                grid[3] + spacing / 2.0,
            ]
        )
        np.testing.assert_allclose(edges, expected, rtol=1e-10)

    def test_edges_interior_are_midpoints(self):
        """Interior edges are always midpoints of adjacent nodes."""
        grid = jnp.array([1.0, 1.5, 3.0])
        edges = edges_for_grid(grid)
        # Interior edges (index 1 and 2)
        expected_interior = jnp.array([(grid[0] + grid[1]) / 2.0, (grid[1] + grid[2]) / 2.0])
        np.testing.assert_allclose(edges[1:3], expected_interior, rtol=1e-10)
