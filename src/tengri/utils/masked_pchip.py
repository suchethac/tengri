# SPDX-License-Identifier: BSD-3-Clause
"""Masked PCHIP lookup: node-exact monotone cubic over populated nodes only (#2861, #2575).

A template grid whose cells are not all populated (for example the MAPPINGS shock
grid, where some (density, B) pairs hold no model) is read here without the
triweight smoother. Along each masked axis the Hermite read uses only populated
nodes:

* the cell is the populated bracket around the query;
* each tangent is the Fritsch-Carlson tangent of the populated subset, computed from
  the nearest populated neighbor on each side (so the stencil is at most four
  populated nodes per axis);
* a query beyond the populated hull holds the nearest populated node (#2575).

On a fully populated mask the read is the plain PCHIP read of
:func:`tengri.utils.grid_interp.interp_nd_pchip` with the axes in that order. A B row
that holds no populated density carries no weight, and the result is NaN only when no
cell is populated at all: the same weighting rule as the triweight masked path it
replaces.

The contraction is separable and reduced in the order velocity, density, B: velocity is
unmasked, density is read within each B row over that row's populated densities, and B
is read across the rows that hold data. PCHIP tangents are nonlinear, so this order is
part of the definition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.utils.grid_interp import (
    _clamp_to_hull,
    _library_axis_query,
    _pchip_edge_slope,
    interp_nd_pchip,
)


def _safe_div(num, den):
    """``num / den`` with a zero result where ``den == 0``, finite in both passes."""
    ok = den != 0.0
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def _secant(x0, x1, y0, y1):
    """Secant slope of one interval, zero for a degenerate interval."""
    return _safe_div(y1 - y0, x1 - x0)


def _interior_tangent(h_prev, h_next, d_left, d_right):
    """Fritsch-Carlson weighted harmonic tangent; zero at a local extremum."""
    w1 = 2.0 * h_next + h_prev
    w2 = h_next + 2.0 * h_prev
    mono = d_left * d_right > 0.0
    d_left_s = jnp.where(mono, d_left, 1.0)
    d_right_s = jnp.where(mono, d_right, 1.0)
    harmonic = _safe_div(w1 + w2, w1 / d_left_s + w2 / d_right_s)
    return jnp.where(mono, harmonic, 0.0)


def _populated_bracket(x, populated, xq):
    """Return the cell ``(a, b)`` and its populated neighbors ``pa``, ``nb``.

    ``b`` is the first populated node above ``xq`` (the last populated node when the
    query sits on or past the populated hull), ``a`` the populated node below it.
    ``pa`` and ``nb`` are the populated nodes beyond the cell. ``pa_ok`` and ``nb_ok``
    flag whether each exists; when one does not, the index falls back to the cell node.
    """
    n = x.shape[0]
    idx = jnp.arange(n)
    b_up = jnp.min(jnp.where(populated & (x > xq), idx, n))
    last = jnp.max(jnp.where(populated, idx, 0))
    b = jnp.where(b_up < n, b_up, last)
    a_raw = jnp.max(jnp.where(populated & (idx < b), idx, -1))
    a_ok = a_raw >= 0
    a = jnp.where(a_ok, a_raw, b)
    pa_raw = jnp.max(jnp.where(populated & (idx < a), idx, -1))
    pa_ok = a_ok & (pa_raw >= 0)
    pa = jnp.where(pa_ok, pa_raw, a)
    nb_raw = jnp.min(jnp.where(populated & (idx > b), idx, n))
    nb_ok = a_ok & (nb_raw < n)
    nb = jnp.where(nb_ok, nb_raw, b)
    return a, b, pa, nb, pa_ok, nb_ok


def _masked_axis_read(x, populated, y, q):
    """One-axis masked PCHIP read of ``y`` (shape ``(n, *rest)``) at ``q``.

    ``populated`` is a boolean ``(n,)`` mask. The tangents are those of the populated
    subset: at a cell node, its interior Fritsch-Carlson tangent uses the populated
    neighbor on its far side, or the one-sided edge rule when that side has no
    populated node. Returns shape ``rest``; zero when no node is populated (the caller
    gives that row no weight).
    """
    has = jnp.any(populated)
    x_lo = jnp.where(has, jnp.min(jnp.where(populated, x, jnp.inf)), x[0])
    x_hi = jnp.where(has, jnp.max(jnp.where(populated, x, -jnp.inf)), x[-1])
    xq = _clamp_to_hull(q, x_lo, x_hi)
    a, b, pa, nb, pa_ok, nb_ok = _populated_bracket(x, populated, xq)

    x0, x1, x2, x3 = x[pa], x[a], x[b], x[nb]
    y0, y1, y2, y3 = y[pa], y[a], y[b], y[nb]
    h12 = x2 - x1
    d01 = _secant(x0, x1, y0, y1)
    d12 = _secant(x1, x2, y1, y2)
    d23 = _secant(x2, x3, y2, y3)

    # Tangent at the lower node a. Interior when a populated node lies below it; else
    # the edge rule when a populated node lies above b; else the secant.
    interior_a = _interior_tangent(jnp.where(pa_ok, x1 - x0, 1.0), h12, d01, d12)
    edge_a = _pchip_edge_slope(h12, jnp.where(nb_ok, x3 - x2, 1.0), d12, d23)
    m1 = jnp.where(pa_ok, interior_a, jnp.where(nb_ok, edge_a, d12))

    # Tangent at the upper node b, mirrored.
    interior_b = _interior_tangent(h12, jnp.where(nb_ok, x3 - x2, 1.0), d12, d23)
    edge_b = _pchip_edge_slope(h12, jnp.where(pa_ok, x1 - x0, 1.0), d12, d01)
    m2 = jnp.where(nb_ok, interior_b, jnp.where(pa_ok, edge_b, d12))

    has_cell = h12 > 0.0
    t = jnp.where(has_cell, (xq - x1) / jnp.where(has_cell, h12, 1.0), 0.0)
    h = jnp.where(has_cell, h12, 0.0)
    t2 = t * t
    t3 = t2 * t
    value = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y1
        + (t3 - 2.0 * t2 + t) * h * m1
        + (-2.0 * t3 + 3.0 * t2) * y2
        + (t3 - t2) * h * m2
    )
    return jnp.where(has, value, 0.0)


def masked_pchip_lookup(grid, axes, point, population_mask, extrapolate_axis0):
    """Masked PCHIP read of a three-axis grid over its populated cells (#2861, #2575).

    Parameters
    ----------
    grid : jnp.ndarray, shape (n_v, n_b, n_density, *trailing)
        Tabulated values. Unpopulated cells may hold any value; they are never read.
    axes : tuple of jnp.ndarray
        Strictly ascending node vectors ``(v, b, density)``.
    point : tuple
        Query ``(v, b, density)``, one scalar per axis.
    population_mask : jnp.ndarray, shape (n_density, n_b)
        1.0 where the cell is populated, 0.0 where it is not.
    extrapolate_axis0 : bool
        Whether the unmasked velocity axis extrapolates its end cubic past its nodes
        (as in :func:`tengri.utils.grid_interp.library_lookup`). The masked axes always
        hold at their populated hull.

    Returns
    -------
    jnp.ndarray, shape (*trailing,)
        Interpolated values. NaN only when no cell is populated.

    Notes
    -----
    Separable, reduced in the order velocity, density, B. Velocity is unmasked, so
    every (B, density) cell has a value after it. Density is then read within each B
    row over that row's populated densities, and B is read across the rows that hold
    any data. A query that lands on an unpopulated (B, density) cell therefore takes
    the populated density bracket of its own B row, not that of a neighboring row.

    PCHIP tangents are nonlinear in the data, so the reduction order is part of the
    definition: on a fully populated mask this equals the unmasked
    :func:`~tengri.utils.grid_interp.interp_nd_pchip` read with the axes ordered
    (velocity, density, B), to rounding. The read is node-exact at populated cells,
    C1 in the query, and uses at most four populated nodes per masked axis. JIT, vmap
    and grad safe: every division is guarded, so the unselected branch is finite in
    the backward pass.

    Raises
    ------
    ValueError
        If the grid is not three-dimensional in its axes or the mask shape is wrong.
    """
    if len(axes) != 3:
        raise ValueError(f"masked lookup expects 3 axes (v, b, density); got {len(axes)}")
    n_b, n_n = grid.shape[1], grid.shape[2]
    if population_mask.shape != (n_n, n_b):
        raise ValueError(
            f"population_mask shape {population_mask.shape} must be (n_density, n_b) = "
            f"({n_n}, {n_b})"
        )
    # Arithmetic in the library default float (float64 under jax_enable_x64), so a float32
    # table is not read back in float32 and a downstream 1e40 scale does not overflow.
    dtype = jnp.result_type(float)
    grid = jnp.asarray(grid, dtype)
    x_v, x_b, x_n = (jnp.asarray(a, dtype) for a in axes)
    q_v, q_b, q_n = (jnp.asarray(p, dtype) for p in point)
    population_mask = jnp.asarray(population_mask, dtype)
    populated = population_mask.T > 0.0  # (n_b, n_density)
    row_valid = jnp.any(populated, axis=1)  # (n_b,)

    # Zero unpopulated cells so every gathered value is finite.
    grid_m = jnp.where(populated.reshape((1, n_b, n_n) + (1,) * (grid.ndim - 3)), grid, 0.0)

    # Velocity: the unmasked PCHIP read at each (B, density) cell.
    qv = _library_axis_query(x_v, q_v, extrapolate_axis0)
    per_v = interp_nd_pchip(grid_m, (x_v,), (qv,), extrapolate=(True,))  # (n_b, n_n, *t)

    # Density: within each B row, over that row's populated densities.
    per_n = jax.vmap(lambda m, y: _masked_axis_read(x_n, m, y, q_n))(populated, per_v)

    # B: across the rows that hold any data.
    out = _masked_axis_read(x_b, row_valid, per_n, q_b)
    return jnp.where(jnp.any(populated), out, jnp.nan)
