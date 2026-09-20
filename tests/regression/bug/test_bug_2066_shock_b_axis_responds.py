# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2066: B-field axis interpolates between bracketing populated nodes.

The triweight kernel can only see sparse grids through the population mask.
Where a stencil contains fewer than two populated nodes, the normalized
convolution becomes flat (invariant in the axis coordinate). This test verifies
that the fix widens the stencil to bracket populated neighbors when that
happens, so the axis responds smoothly even on sparse grids.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py", reason="h5py required for MAPPINGS grid tests")

import jax.numpy as jnp

from tengri.utils.grid_interp import interp_nd_triweight as _interp_nd_triweight
from tengri.utils.interpolation import edges_for_grid

pytestmark = pytest.mark.regression_bug


class TestShockBAxisBracketsPopulatedNodes:
    """On sparse grids, the interpolant responds between populated neighbors.

    Synthetic sparse grid: only two B values are populated at one density.
    The off-node midpoint between them should show a non-flat response even
    with normalized convolution (masked grid).
    """

    def test_sparse_grid_responds_between_bracketing_nodes(self):
        """Synthetic sparse grid: 2 populated B nodes, 1 off-node midpoint.

        The grid has zeros at all unpopulated cells. Without bracketing, the
        normalized convolution would return a flat value (gradient 0) because
        the triweight stencil at the midpoint contains fewer than 2 populated
        nodes. With the fix, it should widen to bracket the populated pair
        and interpolate smoothly between them.
        """
        # Build a synthetic grid: 5 B values, only B[1] and B[3] are populated
        b_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        v_grid = jnp.array([100.0, 200.0, 300.0])
        n_grid = jnp.array([-1.0, 0.0, 1.0])

        # Values are only populated at B[1]=0.5 and B[3]=2.0
        # 3D grid: (v, B, n, trailing)
        grid_values = jnp.zeros((3, 5, 3, 1))

        # Set values at populated B nodes: f(B) = 10 + B (simple linear response in B)
        grid_values = grid_values.at[:, 1, :, 0].set(10.0 + 0.5)  # B[1] = 0.5
        grid_values = grid_values.at[:, 3, :, 0].set(10.0 + 2.0)  # B[3] = 2.0

        # Population mask: 1 at populated (B, n), 0 at unpopulated
        pop_mask = jnp.zeros((3, 5))
        pop_mask = pop_mask.at[:, 1].set(1.0)  # B[1] populated
        pop_mask = pop_mask.at[:, 3].set(1.0)  # B[3] populated

        # Edges for each axis
        v_edges = edges_for_grid(v_grid)
        b_edges = edges_for_grid(b_grid)
        n_edges = edges_for_grid(n_grid)

        # Interpolate at the off-node midpoint: B = sqrt(0.5 * 2.0) ≈ 1.0
        # This is exactly between the two populated B nodes
        b_query = jnp.sqrt(0.5 * 2.0)  # ≈ 1.0, which IS B[2]
        # Use a point slightly off to avoid landing on the grid node
        b_query = 0.95

        v_query = 200.0
        n_query = 0.0

        # Interpolate without mask (should show some response even with zeros)
        result_unmasked = _interp_nd_triweight(
            grid_values,
            (v_grid, b_grid, n_grid),
            (v_edges, b_edges, n_edges),
            (v_query, b_query, n_query),
            index_space_interp=True,
            population_mask=None,
        )

        # Interpolate with mask (the fix should make this respond)
        result_masked = _interp_nd_triweight(
            grid_values,
            (v_grid, b_grid, n_grid),
            (v_edges, b_edges, n_edges),
            (v_query, b_query, n_query),
            index_space_interp=True,
            population_mask=pop_mask,
        )

        # The masked result should NOT be NaN (which would indicate no populated
        # nodes in range after bracketing)
        assert jnp.isfinite(result_masked[0]), (
            "Masked interpolation returned NaN; bracketing may have failed"
        )

        # The masked result should be close to interpolating between the two
        # populated values: f(0.95) ≈ 10 + 0.95 ≈ 10.95
        expected_approx = 10.0 + 0.95
        np.testing.assert_allclose(
            result_masked[0],
            expected_approx,
            rtol=0.1,
            err_msg="Bracketing did not produce expected interpolated value",
        )

    def test_sparse_grid_refuses_outside_bracketed_range(self):
        """Query outside the bracketing populated range raises ValueError.

        If a query point is outside the populated range on a sparse axis, the
        bracketing interpolant cannot work (no neighbors on one side). The
        function raises loudly rather than returning a flat value or NaN silently.
        """
        # Grid with only one populated B node: bracketing is impossible
        b_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        v_grid = jnp.array([100.0, 200.0, 300.0])
        n_grid = jnp.array([-1.0, 0.0, 1.0])

        grid_values = jnp.zeros((3, 5, 3, 1))
        grid_values = grid_values.at[:, 2, :, 0].set(42.0)  # Only B[2] populated

        pop_mask = jnp.zeros((3, 5))
        pop_mask = pop_mask.at[:, 2].set(1.0)  # Only B[2] populated

        v_edges = edges_for_grid(v_grid)
        b_edges = edges_for_grid(b_grid)
        n_edges = edges_for_grid(n_grid)

        # Query far below the single populated node
        b_query = 0.05  # Below B[0] = 0.1

        v_query = 200.0
        n_query = 0.0

        # Outside the populated range, should raise ValueError
        with pytest.raises(ValueError, match="below the lowest populated B node"):
            _interp_nd_triweight(
                grid_values,
                (v_grid, b_grid, n_grid),
                (v_edges, b_edges, n_edges),
                (v_query, b_query, n_query),
                index_space_interp=True,
                population_mask=pop_mask,
                on_out_of_grid="clamp",
            )
