# SPDX-License-Identifier: BSD-3-Clause
"""The local-stencil PCHIP read equals the full-axis read it replaced.

``interp_nd_pchip`` and ``library_lookup`` now gather a 4-node window per axis
and take the tangents from that window. The reference below is the full-axis
tensor-product loop they used before: each axis is reduced by
``_pchip_eval_axis0`` (or ``_linear_eval_axis0``), whose tangents are computed
on the whole axis. The two must agree in value and in gradient to rounding,
at nodes, at the hull edges, in the extrapolation band and beyond it, on
axes of 2 to 9 nodes, and on non-uniform spacing with trailing dimensions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import (
    _library_axis_query,
    _linear_eval_axis0,
    _pchip_eval_axis0,
    interp_nd_pchip,
    library_lookup,
)

pytestmark = pytest.mark.contract

_REL_TOL = 1e-12
_MAX_CELLS = 4_000
_TRAILING_SHAPES = ((), (3,), (2, 4))


def _reference_interp_nd_pchip(grid, axes, point, kinds, extrapolate):
    """Full-axis tensor-product read, as ``interp_nd_pchip`` computed it before."""
    reduced = grid
    for ax, p, kind, ext in zip(axes, point, kinds, extrapolate, strict=True):
        if kind == "pchip":
            reduced = _pchip_eval_axis0(ax, reduced, p, extrapolate=bool(ext))
        else:
            reduced = _linear_eval_axis0(ax, reduced, p)
    return reduced


def _reference_library_lookup(grid, axes, point, extrapolate):
    """Full-axis read of ``library_lookup`` as it was before the local stencil."""
    reduced = grid
    for ax, p, ext in zip(axes, point, extrapolate, strict=True):
        query = _library_axis_query(ax, p, bool(ext))
        reduced = _pchip_eval_axis0(ax, reduced, query, extrapolate=True)
    return reduced


def _draw_axis(rng, n):
    """Strictly ascending, non-uniformly spaced nodes."""
    start = rng.uniform(-3.0, 3.0)
    gaps = rng.uniform(0.05, 2.0, size=n - 1)
    return jnp.asarray(start + np.concatenate([[0.0], np.cumsum(gaps)]))


def _draw_query(rng, x):
    """A query that is a node, an edge, in a band, beyond a band, or interior."""
    x = np.asarray(x)
    gap_lo = x[1] - x[0]
    gap_hi = x[-1] - x[-2]
    choice = rng.integers(0, 7)
    if choice == 0:
        return float(x[rng.integers(len(x))])
    if choice == 1:
        return float(x[0] if rng.random() < 0.5 else x[-1])
    if choice == 2:
        return float(rng.uniform(x[0] - 0.5 * gap_lo, x[0]))
    if choice == 3:
        return float(rng.uniform(x[-1], x[-1] + 0.5 * gap_hi))
    if choice == 4:
        return float(x[0] - rng.uniform(0.6, 2.0) * gap_lo)
    if choice == 5:
        return float(x[-1] + rng.uniform(0.6, 2.0) * gap_hi)
    return float(rng.uniform(x[0], x[-1]))


def _draw_case(rng):
    """One random grid, its axes, kinds, extrapolation flags and query point."""
    ndim = int(rng.integers(2, 7))
    while True:
        lengths = [int(v) for v in rng.integers(2, 10, size=ndim)]
        if np.prod(lengths) <= _MAX_CELLS:
            break
    trailing = _TRAILING_SHAPES[int(rng.integers(0, len(_TRAILING_SHAPES)))]
    axes = tuple(_draw_axis(rng, n) for n in lengths)
    kinds = tuple("linear" if rng.random() < 0.2 else "pchip" for _ in lengths)
    extrapolate = tuple(False if kind == "linear" else bool(rng.random() < 0.5) for kind in kinds)
    grid = jnp.asarray(rng.normal(size=(*lengths, *trailing)))
    point = tuple(_draw_query(rng, ax) for ax in axes)
    return grid, axes, point, kinds, extrapolate, trailing


def _assert_close(got, ref, what):
    got = np.asarray(got)
    ref = np.asarray(ref)
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    err = float(np.max(np.abs(got - ref))) if got.size else 0.0
    assert err <= _REL_TOL * scale, f"{what}: max |diff| {err:.3e} vs scale {scale:.3e}"


def _grad_pair(fn, grid, point, weights):
    """Gradient of ``sum(weights * fn(...))`` wrt the grid and wrt the query."""

    def loss_grid(g):
        return jnp.sum(weights * fn(g, point))

    def loss_point(p):
        return jnp.sum(weights * fn(grid, tuple(p[k] for k in range(len(point)))))

    g_grid = jax.grad(loss_grid)(grid)
    g_point = jax.grad(loss_point)(jnp.asarray(point, dtype=jnp.float64))
    return g_grid, g_point


@pytest.mark.parametrize("seed", range(16))
def test_interp_nd_pchip_local_matches_full_axis(seed):
    """Random 2-D to 6-D grids: values and gradients agree with the full-axis read."""
    rng = np.random.default_rng(1000 + seed)
    grid, axes, point, kinds, extrapolate, trailing = _draw_case(rng)

    got = interp_nd_pchip(grid, axes, point, kinds=kinds, extrapolate=extrapolate)
    ref = _reference_interp_nd_pchip(grid, axes, point, kinds, extrapolate)
    assert got.shape == trailing
    _assert_close(got, ref, "value")

    weights = jnp.asarray(rng.normal(size=trailing))

    def new_fn(g, p):
        return interp_nd_pchip(g, axes, p, kinds=kinds, extrapolate=extrapolate)

    def ref_fn(g, p):
        return _reference_interp_nd_pchip(g, axes, p, kinds, extrapolate)

    got_gg, got_gp = _grad_pair(new_fn, grid, point, weights)
    ref_gg, ref_gp = _grad_pair(ref_fn, grid, point, weights)
    _assert_close(got_gg, ref_gg, "grad wrt grid")
    _assert_close(got_gp, ref_gp, "grad wrt query")


@pytest.mark.parametrize("seed", range(8))
def test_library_lookup_local_matches_full_axis(seed):
    """``library_lookup`` (band-clamped queries, end cubic) agrees with the full-axis read."""
    rng = np.random.default_rng(5000 + seed)
    grid, axes, point, _kinds, _ext, trailing = _draw_case(rng)
    extrapolate = tuple(bool(rng.random() < 0.5) for _ in axes)

    got = library_lookup(grid, axes, point, extrapolate=extrapolate)
    ref = _reference_library_lookup(grid, axes, point, extrapolate)
    assert got.shape == trailing
    _assert_close(got, ref, "value")

    weights = jnp.asarray(rng.normal(size=trailing))

    def new_fn(g, p):
        return library_lookup(g, axes, p, extrapolate=extrapolate)

    def ref_fn(g, p):
        return _reference_library_lookup(g, axes, p, extrapolate)

    got_gg, got_gp = _grad_pair(new_fn, grid, point, weights)
    ref_gg, ref_gp = _grad_pair(ref_fn, grid, point, weights)
    _assert_close(got_gg, ref_gg, "grad wrt grid")
    _assert_close(got_gp, ref_gp, "grad wrt query")


@pytest.mark.parametrize("n", [2, 3])
def test_short_axes_match_full_axis_at_every_node(n):
    """Two- and three-node axes (the whole axis is the window) match the full-axis read."""
    rng = np.random.default_rng(77 + n)
    x = _draw_axis(rng, n)
    y = _draw_axis(rng, 4)
    axes = (x, y)
    grid = jnp.asarray(rng.normal(size=(n, 4, 2)))
    kinds = ("pchip", "pchip")
    extrapolate = (False, True)
    for i, xq in enumerate(np.asarray(x)):
        for j, yq in enumerate(np.asarray(y)):
            point = (float(xq), float(yq))
            got = interp_nd_pchip(grid, axes, point, kinds=kinds, extrapolate=extrapolate)
            ref = _reference_interp_nd_pchip(grid, axes, point, kinds, extrapolate)
            _assert_close(got, ref, f"n={n} node {point}")
            _assert_close(got, grid[i, j], f"n={n} stored node {point}")
