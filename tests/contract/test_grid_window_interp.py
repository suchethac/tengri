# SPDX-License-Identifier: BSD-3-Clause
"""Contract for the windowed triweight contraction (``compute_grid_window``).

``compute_grid_weights`` returns a DENSE weight vector over the whole axis,
and callers contract it against the table with ``tensordot``/``einsum``.  On
the small axes (metallicity ~10 nodes, CLOUDY ionization ~10) that is free.
On the WavePrecomp redshift axis it is not: ``n_z=250`` multiplied against a
``(n_z, n_met, n_age, n_filt)`` table dominates the free-redshift gradient,
even though the triweight kernel has support on |z| < 3*scatter — about five
nodes out of the 250.

This module pins the two properties a windowed replacement must have:

1. it agrees with the dense contraction (the excluded weights are EXACTLY
   zero, so this is an identity, not an approximation), and
2. its cost does not grow with the length of the axis.

Property 2 is the point of the change, so it is asserted directly, with the
dense contraction alongside as a control — a cost instrument that cannot see
the n_z scaling of the OLD code cannot be trusted to certify its absence in
the new code.

Property 2 is about the COMPILED GRAPH, not the clock. FLOPs and bytes are
exact and machine-independent, which is why they are asserted here; they are
also a poor predictor of time. Measured end to end, this change removed 6.8x
of the free-redshift gradient's FLOPs and 5.1x of its ``bytes accessed`` and
bought about 1.9x of wall clock — because ``bytes accessed`` counts logical
operand bytes per op, not DRAM traffic after XLA fusion. Do not convert the
numbers this test asserts into a speedup; measure one.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def _dense_reference(x, grid, table, scatter):
    """The contraction as it is written today, for equivalence checking."""
    import jax.numpy as jnp

    from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

    w = compute_grid_weights(x, grid, scatter=scatter, edges=edges_for_grid(grid))
    return jnp.tensordot(w, table, axes=([0], [0]))


def _uniform_grid(n, lo=0.5, hi=12.0):
    """A WavePrecomp-shaped redshift axis."""
    import jax.numpy as jnp

    return jnp.linspace(lo, hi, n)


# Query points chosen to exercise every branch: exactly on a node, mid-cell,
# at the point where the nearest-node index flips (the window slides by one —
# the value must stay continuous there), on both boundary nodes, and outside
# the grid at both ends where the kernel misses entirely and the fallback fires.
def _query_points(grid):
    g = np.asarray(grid)
    dz = float(g[1] - g[0])
    return {
        "on_node_interior": float(g[100]),
        "mid_cell": float(g[100]) + 0.5 * dz,
        "just_below_flip": float(g[100]) + 0.4999 * dz,
        "just_above_flip": float(g[100]) + 0.5001 * dz,
        "quarter_cell": float(g[57]) + 0.25 * dz,
        "first_node": float(g[0]),
        "second_node": float(g[1]),
        "last_node": float(g[-1]),
        "penultimate_node": float(g[-2]),
        "below_grid": float(g[0]) - 3.0 * dz,
        "above_grid": float(g[-1]) + 3.0 * dz,
    }


@pytest.fixture(scope="module")
def table():
    """A (n_z, n_met, n_age, n_filt)-shaped stand-in for ``ssp_phot_table``."""
    rng = np.random.default_rng(0)
    return jax.numpy.asarray(rng.normal(size=(250, 5, 7, 4)))


def test_window_matches_dense_contraction(table):
    """The windowed contraction reproduces the dense one at every query point.

    The nodes the window drops carry weights that are exactly zero (the
    triweight CDF clamps outside |z| > 3), so this is an identity. The only
    admissible difference is float summation order in the normalization,
    which is why the tolerance is 1e-12 and not 0.
    """
    from tengri.utils.interpolation import apply_grid_window, compute_grid_window

    grid = _uniform_grid(250)
    dz = float(np.asarray(grid)[1] - np.asarray(grid)[0])
    scatter = 0.5 * dz

    for name, x in _query_points(grid).items():
        want = np.asarray(_dense_reference(x, grid, table, scatter))
        start, w = compute_grid_window(x, grid, bandwidth_cells=0.5)
        # The tengri convention scatter = dz/2 gives a five-node window. Pinned
        # because the width sets the returned shape and so the compile-cache
        # key: a silent widen to 7 would be a 40% cost regression nobody sees.
        assert w.shape[0] == 5, f"expected a 5-node window at {name}, got {w.shape[0]}"
        got = np.asarray(apply_grid_window(table, start, w))
        np.testing.assert_allclose(
            got, want, rtol=1e-12, atol=0.0, err_msg=f"windowed != dense at {name} (z={x})"
        )


def test_window_weights_match_dense_weights_in_place(table):
    """The window's weights are the dense weights, sliced — not a renormalization.

    Guards the failure mode where the window drops a node with nonzero weight
    and the renormalization hides it: the result would still sum to 1 and
    still look plausible, but would be wrong by the dropped weight.
    """
    from tengri.utils.interpolation import (
        compute_grid_weights,
        compute_grid_window,
        edges_for_grid,
    )

    grid = _uniform_grid(250)
    dz = float(np.asarray(grid)[1] - np.asarray(grid)[0])
    scatter = 0.5 * dz

    edges = edges_for_grid(grid)
    for name, x in _query_points(grid).items():
        dense = np.asarray(compute_grid_weights(x, grid, scatter=scatter, edges=edges))
        start, w = compute_grid_window(x, grid, bandwidth_cells=0.5)
        s, width = int(start), int(np.asarray(w).shape[0])
        np.testing.assert_allclose(
            np.asarray(w), dense[s : s + width], rtol=1e-12, atol=0.0, err_msg=f"weights at {name}"
        )
        # Everything outside the window must be exactly zero, or the window
        # is too narrow and the agreement above is luck.
        outside = np.concatenate([dense[:s], dense[s + width :]])
        n_lost = np.count_nonzero(outside)
        assert n_lost == 0, f"{name}: {n_lost} nonzero weights dropped by the window"


def test_window_gradient_matches_dense(table):
    """d/dz through the window equals d/dz through the dense contraction.

    The window start is an integer function of z with zero tangent, so the
    gradient must flow through the weights alone — identical to the dense path.
    """
    import jax.numpy as jnp

    from tengri.utils.interpolation import apply_grid_window, compute_grid_window

    grid = _uniform_grid(250)
    dz = float(np.asarray(grid)[1] - np.asarray(grid)[0])
    scatter = 0.5 * dz

    def dense_scalar(x):
        return jnp.sum(_dense_reference(x, grid, table, scatter))

    def window_scalar(x):
        start, w = compute_grid_window(x, grid, bandwidth_cells=0.5)
        return jnp.sum(apply_grid_window(table, start, w))

    for name, x in _query_points(grid).items():
        want = float(jax.grad(dense_scalar)(x))
        got = float(jax.grad(window_scalar)(x))
        # Absolute floor: at a few query points the dense gradient is near
        # zero by symmetry, where a relative tolerance is meaningless.
        np.testing.assert_allclose(
            got, want, rtol=1e-10, atol=1e-10 * max(1.0, abs(want)), err_msg=f"grad at {name}"
        )


def test_window_is_continuous_where_it_slides(table):
    """Straddling the index flip changes the window but not the value.

    When z crosses a cell midpoint the window slides by one node: one node
    leaves and another enters. Both carry exactly zero weight at that instant,
    so the interpolant stays continuous. If the window were too narrow this is
    where a jump would appear — and a jump here would put a cliff in the
    likelihood that NUTS would hit thousands of times per fit.

    Asserted against a control, not against constancy: the interpolant has a
    genuine nonzero slope, so two distinct query points are never equal. The
    control is a step of the SAME size that does not cross a slide. Verdict
    rule, fixed before running: dropping a supported node would jump the value
    by (dropped weight) x (table scale) ~ 1e-2, which is ~1e6 x the smooth
    step, so a 10x ratio separates the two outcomes with orders of magnitude
    to spare.
    """
    from tengri.utils.interpolation import apply_grid_window, compute_grid_window

    grid = _uniform_grid(250)
    dz = float(np.asarray(grid)[1] - np.asarray(grid)[0])
    scatter = 0.5 * dz
    flip = float(np.asarray(grid)[100]) + 0.5 * dz

    eps = 1e-9 * dz

    def probe(x):
        start, w = compute_grid_window(x, grid, bandwidth_cells=0.5)
        return int(start), np.asarray(apply_grid_window(table, start, w))

    s_lo, v_lo = probe(flip - eps)  # crossing pair
    s_hi, v_hi = probe(flip + eps)
    s_c0, v_c0 = probe(flip + eps)  # control pair: same step, no slide
    s_c1, v_c1 = probe(flip + 3.0 * eps)

    assert s_lo != s_hi, "query points do not straddle the window slide; test is vacuous"
    assert s_c0 == s_c1, "control pair straddles a slide; it is not a control"

    crossing = float(np.max(np.abs(v_hi - v_lo)))
    control = float(np.max(np.abs(v_c1 - v_c0)))
    assert crossing < 10.0 * control + 1e-14, (
        f"value jumps across the window slide: {crossing:.3e} over the same step that moves "
        f"it {control:.3e} without sliding ({crossing / max(control, 1e-300):.1f}x) — the "
        f"window is dropping weight it should carry"
    )


def test_non_uniform_grid_is_rejected():
    """A non-uniform axis raises rather than silently returning a wrong window.

    The O(1) index arithmetic assumes uniform spacing. Falling back to a
    dense argmin would be a silent performance regression; returning the
    wrong window would be a silent numerical one. Neither is acceptable.
    """
    import jax.numpy as jnp

    from tengri.utils.interpolation import compute_grid_window

    grid = jnp.asarray([0.0, 0.1, 0.3, 0.7, 1.5])
    with pytest.raises(ValueError, match="uniform"):
        compute_grid_window(0.4, grid)


@pytest.mark.parametrize("n_small,n_large", [(50, 500)])
def test_window_cost_is_independent_of_axis_length(n_small, n_large):
    """The compiled cost stops scaling with the grid length.

    FLOP counts come from the compiled executable, so this assertion is a
    property of the graph and is immune to machine load — unlike a wall-clock
    ratio, which on a contended box moved 2.7x for an unchanged arm.

    The dense arm is the control: it must SHOW the scaling this test claims
    the windowed arm removes.
    """
    import jax.numpy as jnp

    from tengri.utils.interpolation import apply_grid_window, compute_grid_window

    def flops(fn, *args):
        cost = jax.jit(fn).lower(*args).compile().cost_analysis()
        if isinstance(cost, list):
            cost = cost[0] if cost else {}
        return float(dict(cost or {}).get("flops", 0.0))

    def measure(n):
        grid = _uniform_grid(n)
        dz = float(np.asarray(grid)[1] - np.asarray(grid)[0])
        scatter = 0.5 * dz
        rng = np.random.default_rng(1)
        tab = jnp.asarray(rng.normal(size=(n, 5, 7, 4)))

        def dense(x, tab):
            return _dense_reference(x, grid, tab, scatter)

        def windowed(x, tab):
            start, w = compute_grid_window(x, grid, bandwidth_cells=0.5)
            return apply_grid_window(tab, start, w)

        x = float(np.asarray(grid)[n // 2]) + 0.3 * dz
        return flops(dense, x, tab), flops(windowed, x, tab)

    dense_small, win_small = measure(n_small)
    dense_large, win_large = measure(n_large)

    dense_ratio = dense_large / dense_small
    win_ratio = win_large / win_small

    # Control: the instrument must see the 10x axis growth in the old code.
    assert dense_ratio > 5.0, (
        f"cost instrument does not see n_z in the DENSE path "
        f"({dense_small:.0f} -> {dense_large:.0f}, {dense_ratio:.2f}x); "
        f"the windowed assertion below would be vacuous"
    )
    # The claim: windowed cost is flat in n_z.
    assert win_ratio < 1.5, (
        f"windowed cost still scales with n_z "
        f"({win_small:.0f} -> {win_large:.0f}, {win_ratio:.2f}x)"
    )
