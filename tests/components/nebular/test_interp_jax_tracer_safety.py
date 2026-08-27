# SPDX-License-Identifier: BSD-3-Clause
"""Test JAX tracer safety for interpolation functions.

Verifies that _interp_index_weight uses jnp.take for safe indexing with
traced values under vmap, fixing the TracerArrayConversionError that
occurred when direct numpy-style indexing was used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import _interp_index_weight

pytestmark = pytest.mark.regression_bug


def test_interp_index_weight_vmap_safety():
    """_interp_index_weight is JAX-safe when called under vmap on traced values.

    Regression test for JAX tracer bug (issue found while wiring #2070): direct indexing
    grid[idx + 1] fails on traced values. Using jnp.take() fixes this.
    """
    # Create a simple grid
    grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Test values to interpolate: some in-grid, one at boundary
    query_values = jnp.array([0.5, 1.5, 2.5, 3.5, 0.0, 4.0])

    # Compute under vmap (this will trace the indices)
    def compute_weights(x):
        idx, w = _interp_index_weight(x, grid)
        return idx, w

    # This should work without TracerArrayConversionError
    vmapped_compute = jax.vmap(compute_weights)
    indices, w = vmapped_compute(query_values)

    # Verify shapes
    assert indices.shape == (6,), f"Expected (6,), got {indices.shape}"
    assert w.shape == (6,), f"Expected (6,), got {w.shape}"

    # Verify values are in expected ranges
    assert jnp.all(indices >= 0) and jnp.all(indices < len(grid) - 1)
    assert jnp.all(w >= 0.0) and jnp.all(w <= 1.0)

    # Verify specific results: for x=0.5 between grid[0]=0 and grid[1]=1,
    # we should get idx=0, w=0.5
    assert indices[0] == 0, f"Expected idx=0 for x=0.5, got {indices[0]}"
    assert jnp.allclose(w[0], 0.5), f"Expected w=0.5 for x=0.5, got {w[0]}"


def test_interp_index_weight_vmap_matches_loop():
    """_interp_index_weight under vmap matches a Python loop of scalar calls.

    This verifies that the vectorized computation via vmap produces bit-exact
    results matching sequential scalar calls, confirming the tracer fix
    maintains numerical consistency.
    """
    # Create a random grid and query points
    np.random.seed(42)
    grid = jnp.array(np.sort(np.random.uniform(0, 10, 8)))
    query_values = jnp.array(np.random.uniform(-1, 11, 10))

    # Compute under vmap
    def compute_weights(x):
        idx, w = _interp_index_weight(x, grid)
        return idx, w

    vmapped_compute = jax.vmap(compute_weights)
    indices_vmap, weights_vmap = vmapped_compute(query_values)

    # Compute via loop (scalar calls)
    indices_loop = []
    weights_loop = []
    for x in query_values:
        idx, w = _interp_index_weight(x, grid)
        indices_loop.append(idx)
        weights_loop.append(w)

    indices_loop = jnp.array(indices_loop)
    weights_loop = jnp.array(weights_loop)

    # Compare: vmap and loop should be bit-exact
    assert jnp.array_equal(indices_vmap, indices_loop), (
        f"Indices differ: vmap={indices_vmap}, loop={indices_loop}"
    )
    assert jnp.allclose(weights_vmap, weights_loop, rtol=1e-15, atol=0), (
        f"Weights differ: vmap={weights_vmap}, loop={weights_loop}"
    )


def test_interp_index_weight_boundary_handling_under_vmap():
    """_interp_index_weight handles boundary clipping correctly under vmap.

    The function should clip out-of-grid queries to valid bounds before
    computing indices and weights. This test verifies the behavior is
    preserved when traced.
    """
    grid = jnp.array([0.0, 1.0, 2.0, 3.0])

    # Query values outside grid: < 0, at 0, at max, > max
    query_values = jnp.array([-5.0, 0.0, 3.0, 10.0])

    def compute_weights(x):
        idx, w = _interp_index_weight(x, grid)
        return idx, w

    vmapped_compute = jax.vmap(compute_weights)
    indices, _w = vmapped_compute(query_values)

    # All queries should map to valid indices in [0, len(grid)-2]
    assert jnp.all(indices >= 0), f"Indices have negative values: {indices}"
    assert jnp.all(indices < len(grid) - 1), f"Indices out of range: {indices}"

    # Boundary behavior:
    # x=-5.0 should be clipped to grid[0]=0.0, so idx=0, w=0
    # x=0.0 should be at grid[0], so idx=0, w=0
    # x=3.0 should be at grid[-1], so idx=len(grid)-2, w=1.0 (clipped)
    # x=10.0 should be clipped to grid[-1]=3.0, so idx=len(grid)-2, w=1.0 (clipped)
    assert indices[0] == 0, f"Expected idx=0 for x=-5, got {indices[0]}"
    assert indices[1] == 0, f"Expected idx=0 for x=0, got {indices[1]}"
    assert indices[2] == len(grid) - 2, f"Expected idx={len(grid) - 2} for x=3, got {indices[2]}"
    assert indices[3] == len(grid) - 2, f"Expected idx={len(grid) - 2} for x=10, got {indices[3]}"
