# SPDX-License-Identifier: BSD-3-Clause
"""Test JAX tracer safety for interpolation functions.

Verifies that _interp_index_weight uses jnp.take for safe indexing with
traced values under vmap, fixing the TracerArrayConversionError that
occurred when direct numpy-style indexing was used on numpy arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import _interp_index_weight

pytestmark = pytest.mark.regression_bug


def test_interp_index_weight_numpy_array_vmap_safety():
    """_interp_index_weight is JAX-safe with numpy arrays under vmap.

    Regression test for JAX tracer bug (found while wiring #2070): direct
    indexing grid[idx + 1] fails when grid is a numpy array and idx is a
    traced value. Using jnp.take() fixes this. The MAPPINGS loader passes
    numpy axes (as the loader does), so this test pins the actual bug.
    """
    # Create a NUMPY array (as _load_stellar_grid does with the axes)
    grid = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

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


def test_interp_index_weight_jnp_array_vmap_parity():
    """_interp_index_weight works with jnp arrays under vmap (parity test).

    Confirms that the fix also works when grid is a jnp array (both numpy and
    jnp arrays should work; we test both to ensure coverage).
    """
    # Create a JAX NUMPY array (for parity — works but not the bug path)
    grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    query_values = jnp.array([0.5, 1.5, 2.5, 3.5])

    def compute_weights(x):
        idx, w = _interp_index_weight(x, grid)
        return idx, w

    vmapped_compute = jax.vmap(compute_weights)
    indices, w = vmapped_compute(query_values)

    # Verify it works
    assert indices.shape == (4,), f"Expected (4,), got {indices.shape}"
    assert w.shape == (4,), f"Expected (4,), got {w.shape}"
    assert jnp.all(indices >= 0) and jnp.all(indices < len(grid) - 1)
    assert jnp.all(w >= 0.0) and jnp.all(w <= 1.0)


def test_interp_index_weight_numpy_matches_scalar():
    """_interp_index_weight under vmap with numpy grid matches scalar calls.

    Verifies that vectorized computation via vmap produces bit-exact results
    matching sequential scalar calls, confirming the tracer fix maintains
    numerical consistency when grid is a numpy array.
    """
    # Create a numpy grid (as the loader does)
    np.random.seed(42)
    grid_np = np.asarray(np.sort(np.random.uniform(0, 10, 8)), dtype=np.float64)
    query_values = jnp.array(np.random.uniform(-1, 11, 10))

    # Compute under vmap
    def compute_weights(x):
        idx, w = _interp_index_weight(x, grid_np)
        return idx, w

    vmapped_compute = jax.vmap(compute_weights)
    indices_vmap, weights_vmap = vmapped_compute(query_values)

    # Compute via loop (scalar calls)
    indices_loop = []
    weights_loop = []
    for x in query_values:
        idx, w = _interp_index_weight(x, grid_np)
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
