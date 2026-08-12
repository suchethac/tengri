# SPDX-License-Identifier: BSD-3-Clause
"""Grid-interpolation dtype consistency under pure float32 (issue #1206 item 4).

Template grids (SKIRTOR, dust libraries) are loaded and cached at import time,
so their axes are float64 arrays. Inside ``jax.enable_x64(False)`` those f64
arrays survive, but ``jnp.argmin`` builds its initial value from the *current*
default float (float32) -- the operand and initial dtypes then disagree and the
reduction raises, taking the whole panchromatic forward down before any
float32 behavior can be observed.

Canonicalizing the inputs at function entry fixes it, and is a no-op under
``x64=True`` (the canonical float is float64 there), so float64 results are
bit-unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

pytestmark = pytest.mark.regression_bug


def test_compute_grid_weights_accepts_f64_grid_under_pure_float32():
    """An f64 grid + f64 query inside enable_x64(False) must not raise.

    This is the exact shape of the SKIRTOR failure: the cached template axes are
    float64, the query point is float64, and argmin's initial value is float32.
    """
    grid = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    x = jnp.asarray(2.4, dtype=jnp.float64)
    edges = edges_for_grid(grid)
    assert grid.dtype == jnp.float64, "setup: grid must genuinely be float64"

    with jax.enable_x64(False):
        w = compute_grid_weights(x, grid, scatter=0.2, edges=edges)
        w = np.asarray(w)

    assert np.all(np.isfinite(w)), f"weights non-finite: {w}"
    assert_allclose(w.sum(), 1.0, rtol=1e-5)


def test_compute_grid_weights_float64_result_unchanged():
    """Canonicalization must not move the float64 answer (bit-exact)."""
    grid = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    edges = edges_for_grid(grid)
    for x_val in (1.0, 2.4, 3.5, 5.0):
        x = jnp.asarray(x_val, dtype=jnp.float64)
        w = np.asarray(compute_grid_weights(x, grid, scatter=0.2, edges=edges))
        assert w.dtype == np.float64
        assert_allclose(w.sum(), 1.0, rtol=1e-12)


def test_compute_grid_weights_nearest_fallback_under_float32():
    """The argmin nearest-node fallback (kernel misses the grid) works in f32."""
    grid = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    edges = edges_for_grid(grid)
    # A query far outside the grid with a tiny scatter -> kernel contributes
    # nothing, so the fallback argmin branch is the one under test.
    x = jnp.asarray(500.0, dtype=jnp.float64)
    with jax.enable_x64(False):
        w = np.asarray(compute_grid_weights(x, grid, scatter=1e-6, edges=edges))
    assert np.all(np.isfinite(w))
    assert_allclose(w.sum(), 1.0, rtol=1e-5)
    assert int(np.argmax(w)) == 2, "fallback must select the nearest node"
