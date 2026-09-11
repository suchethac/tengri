# SPDX-License-Identifier: BSD-3-Clause
"""``compute_grid_weights`` must evaluate the triweight CDF once per edge.

Bin ``k``'s upper edge is bin ``k+1``'s lower edge. Evaluating ``edges[:-1]``
and ``edges[1:]`` as two separate kernel calls therefore computes every
interior edge twice, from the same expression and the same operands — ``2n``
evaluations against ``n + 1`` distinct edges.

Differencing adjacent entries of a single evaluation feeds the subtraction the
identical pair, so the weights are unchanged; measured bit-identical across
33,600 elements. Measured FLOPs of a vmapped call over 4096 query points:

===========  ==========  ==========  ======
grid nodes   two-slice   shared      ratio
===========  ==========  ==========  ======
11             2.011 M     1.397 M   1.44x
32             5.882 M     3.977 M   1.48x
64            11.780 M     7.909 M   1.49x
===========  ==========  ==========  ======

This is the same defect that :mod:`tests.regression.test_lgmet_weights_parcels`
guards in the metallicity MDF. It is guarded separately here because this
utility serves a different set of callers — SKIRTOR and the dust template
grids — and a future edit could reintroduce it in one place without the other.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.interpolation import (
    compute_grid_weights,
    edges_for_grid,
    tw_cuml_kern,
)

pytestmark = pytest.mark.regression_bug


def _two_slice(x, grid, scatter=0.2):
    """The superseded form: one kernel call per bin *side*."""
    edges = edges_for_grid(grid)
    raw = tw_cuml_kern(x, edges[:-1], scatter) - tw_cuml_kern(x, edges[1:], scatter)
    total = jnp.sum(raw)
    nearest = jnp.argmin(jnp.abs(grid - x))
    fallback = jnp.zeros(grid.shape[0]).at[nearest].set(1.0)
    return jnp.where(total > 0.0, raw / total, fallback)


@pytest.mark.parametrize("n_nodes", [3, 5, 11, 32, 64])
@pytest.mark.parametrize("scatter", [0.05, 0.2, 1.0])
def test_shared_edge_form_is_bit_identical(n_nodes, scatter):
    """Same weights, to the last bit, across grid sizes and kernel widths."""
    grid = jnp.linspace(-3.0, 1.0, n_nodes)
    rng = np.random.default_rng(n_nodes * 17 + int(scatter * 100))
    # In range, off both ends, and exactly on nodes.
    xs = np.concatenate([rng.uniform(-4.5, 2.5, 200), np.asarray(grid), [-50.0, 50.0]])

    for x in xs:
        want = np.asarray(_two_slice(jnp.asarray(x), grid, scatter))
        got = np.asarray(compute_grid_weights(jnp.asarray(x), grid, scatter))
        assert np.array_equal(got, want), (
            f"x={x}, n={n_nodes}, scatter={scatter}: max |diff| {np.abs(got - want).max():.3e}"
        )


def test_weights_are_normalized_and_non_negative():
    """Physical contract, independent of the reference implementation."""
    grid = jnp.linspace(-3.0, 1.0, 15)
    rng = np.random.default_rng(3)

    for x in rng.uniform(-5.0, 3.0, 200):
        w = compute_grid_weights(jnp.asarray(x), grid)
        assert bool(jnp.all(w >= 0.0)), f"negative weight at x={x}"
        np.testing.assert_allclose(float(jnp.sum(w)), 1.0, rtol=1e-12)


def test_off_grid_query_falls_back_to_the_nearest_node():
    """Far outside the grid every bin underflows; the fallback must engage."""
    grid = jnp.linspace(-3.0, 1.0, 15)

    lo = compute_grid_weights(jnp.asarray(-50.0), grid)
    hi = compute_grid_weights(jnp.asarray(50.0), grid)

    assert int(jnp.argmax(lo)) == 0
    assert int(jnp.argmax(hi)) == grid.shape[0] - 1
    np.testing.assert_allclose(float(jnp.sum(lo)), 1.0, rtol=1e-12)
    np.testing.assert_allclose(float(jnp.sum(hi)), 1.0, rtol=1e-12)


def test_does_fewer_kernel_evaluations_than_the_two_slice_form():
    """The point of the change: strictly less arithmetic, same answer.

    Asserted as a ratio rather than an absolute count so it tracks XLA
    version changes, but fails if the redundancy is reintroduced.
    """
    grid = jnp.linspace(-3.0, 1.0, 32)
    xs = jnp.linspace(-3.5, 1.5, 4096)

    def flops(fn):
        c = jax.jit(jax.vmap(lambda x: fn(x, grid))).lower(xs).compile().cost_analysis()
        if isinstance(c, list):
            c = c[0]
        return c["flops"]

    two_slice = flops(_two_slice)
    shared = flops(compute_grid_weights)

    assert shared < two_slice, f"shared-edge form is not cheaper: {shared} vs {two_slice} FLOPs"
    assert two_slice / shared > 1.2, (
        f"expected a clear reduction, got {two_slice / shared:.3f}x — "
        "has the redundancy crept back?"
    )


def test_gradients_flow_and_are_finite():
    """These weights sit under template interpolation on the fit path."""
    grid = jnp.linspace(-3.0, 1.0, 15)

    g = jax.grad(lambda x: jnp.sum(compute_grid_weights(x, grid) ** 2))(jnp.asarray(-1.2))

    assert bool(jnp.isfinite(g)), "non-finite gradient through the grid weights"
    assert jnp.any(g != 0.0), (
        "`g` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
