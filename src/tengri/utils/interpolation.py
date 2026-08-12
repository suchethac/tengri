# SPDX-License-Identifier: BSD-3-Clause
"""Smooth grid interpolation via the triweight kernel (Hearin et al. 2023).

Provides C²-continuous weights for interpolating over 1-D grid axes.
Used by the CLOUDY nebular grid interpolation and available for any
future grid-based model component.

Note: the DSPS/SPS metallicity interpolation in ``dsps_wrapper.py``
keeps its own triweight kernel — that code mirrors upstream DSPS exactly
and should not be replaced.

References
----------

- Hearin et al. 2023, Open J. Astrophysics, 6, 1 (DSPS)

"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes as jax_dtypes


@jax.jit
def tw_cuml_kern(x: float, m: float, h: float) -> float:
    """Triweight kernel complementary CDF at *m* for a kernel centered at *x*.

    Evaluates the CDF of the triweight kernel (support |z| < 3) using
    Horner's method for numerical stability.

    ``compute_grid_weights(x, grid, h)[i]`` equals
    ``tw_cuml_kern(x, edge_lo[i], h) - tw_cuml_kern(x, edge_hi[i], h)``

    Parameters
    ----------
    x : float
        Query point (kernel center).
    m : float or array
        Location(s) at which to evaluate the CDF.
    h : float
        Kernel bandwidth (same units as *x* and *m*).

    Returns
    -------
    float or array
        CDF value(s) in [0, 1].
    """
    z = (x - m) / h
    z2 = z * z
    # Horner form: fewer FLOPs, better numerical stability than separate pow()
    val = (
        z * (35.0 / 96.0 + z2 * (-35.0 / 864.0 + z2 * (7.0 / 2592.0 + z2 * (-5.0 / 69984.0))))
        + 0.5
    )
    val = jnp.where(z < -3.0, 0.0, val)
    val = jnp.where(z > 3.0, 1.0, val)
    return val


def edges_for_grid(grid: jnp.ndarray) -> jnp.ndarray:
    """Compute bin edges from grid midpoints for triweight interpolation.

    Returns ``n + 1`` edges bracketing ``n`` grid nodes.  Interior edges are
    midpoints between adjacent nodes.  Boundary edges use half-spacing
    beyond the outermost nodes.

    Precompute once per static grid and pass to :func:`compute_grid_weights`
    to avoid rebuilding inside a JIT trace.

    Parameters
    ----------
    grid : array, shape (n,)
        Sorted grid node values (ascending).

    Returns
    -------
    array, shape (n + 1,)
        Bin edges.
    """
    half_lo = (grid[1] - grid[0]) / 2.0
    half_hi = (grid[-1] - grid[-2]) / 2.0
    return jnp.concatenate([grid[:1] - half_lo, (grid[:-1] + grid[1:]) / 2.0, grid[-1:] + half_hi])


def compute_grid_weights(
    x: float,
    grid: jnp.ndarray,
    scatter: float = 0.2,
    edges: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Smooth triweight-kernel weights over a 1-D grid axis.

    Integrates the triweight kernel CDF between bin edges — the same approach
    as DSPS's ``triweighted_histogram`` — and returns a weight vector that
    sums to 1.  Unlike piecewise-linear interpolation the weights transition
    smoothly through grid nodes, giving C²-continuous gradients.

    Parameters
    ----------
    x : float
        Query point on the axis.
    grid : array, shape (n,)
        Sorted grid node values (ascending).
    scatter : float
        Kernel bandwidth (same units as ``grid``).  Smaller values concentrate
        weight near the nearest node; larger values spread across more bins.
        Default 0.2, consistent with the DSPS lgmet_scatter convention.
    edges : array, shape (n + 1,) or None
        Precomputed bin edges from :func:`edges_for_grid`.  When ``None``
        (default), edges are computed on the fly.

    Returns
    -------
    array, shape (n,)
        Non-negative weights summing to 1.
    """
    # Canonicalize to the dtype JAX is currently defaulting to. Template grids
    # (SKIRTOR, dust libraries) are built and cached at import, so their axes stay
    # float64 even inside ``jax.enable_x64(False)``; the ``jnp.argmin`` below then
    # builds a float32 initial value against a float64 operand and the reduction
    # raises. Under x64=True the canonical float IS float64, so this is a no-op and
    # float64 results are bit-unchanged (#1206).
    dt = jax_dtypes.canonicalize_dtype(jnp.result_type(jnp.asarray(x), jnp.asarray(grid)))
    x = jnp.asarray(x, dtype=dt)
    grid = jnp.asarray(grid, dtype=dt)

    n = grid.shape[0]
    if edges is None:
        edges = edges_for_grid(grid)
    else:
        edges = jnp.asarray(edges, dtype=dt)
    raw = tw_cuml_kern(x, edges[:-1], scatter) - tw_cuml_kern(x, edges[1:], scatter)
    total = jnp.sum(raw)
    # Fallback: place all weight on the nearest bin if kernel misses the grid
    nearest = jnp.argmin(jnp.abs(grid - x))
    fallback = jnp.zeros(n).at[nearest].set(1.0)
    return jnp.where(total > 0.0, raw / total, fallback)


def _check_uniform(grid: jnp.ndarray) -> None:
    """Raise if *grid* is a concrete, non-uniform or descending axis.

    Skipped for a traced grid, where the spacings are not available at trace
    time.  Every grid tengri builds is a ``linspace`` and reaches this check
    concrete, so the guard is live on the real path rather than decorative.

    Parameters
    ----------
    grid : array, shape (n,)
        Candidate grid node values.
    """
    try:
        g = np.asarray(grid, dtype=float)
    except Exception:  # a JAX tracer cannot be converted — nothing to check
        return
    if g.ndim != 1 or g.size < 2:
        raise ValueError(f"grid must be 1-D with at least 2 nodes; got shape {g.shape}")
    d = np.diff(g)
    if d[0] <= 0.0 or not np.allclose(d, d[0], rtol=1e-9, atol=0.0):
        raise ValueError(
            "compute_grid_window requires a uniform ascending grid; got spacings in "
            f"[{d.min():.6g}, {d.max():.6g}]. Use compute_grid_weights for a non-uniform axis."
        )


def compute_grid_window(
    x: float,
    grid: jnp.ndarray,
    bandwidth_cells: float = 0.5,
    edges: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kernel-supported slice of :func:`compute_grid_weights` on a uniform grid.

    :func:`compute_grid_weights` returns a dense weight vector over the whole
    axis, but the triweight kernel has support only on ``|x - m| < 3 * scatter``.
    On a uniform grid that is a fixed number of adjacent nodes regardless of how
    fine the grid is, so the dense vector is mostly exact zeros.  This function
    returns just the supported window — a start index and the weights at
    ``grid[start : start + 2 * half_width + 1]`` — for use with
    :func:`apply_grid_window`.

    The dropped weights are **exactly** zero (:func:`tw_cuml_kern` clamps the
    CDF outside ``|z| > 3``), so this is an algebraic identity, not an
    approximation: the interpolated value, its gradient, and its C²-continuity
    are all unchanged.  What changes is cost — contracting a ``(n, ...)`` table
    over the window instead of over ``n`` makes the contraction independent of
    the grid length.

    Parameters
    ----------
    x : float
        Query point on the axis (same units as ``grid``).
    grid : array, shape (n,)
        Uniformly spaced, ascending grid node values.  May be traced; only its
        shape is needed at trace time.
    bandwidth_cells : float
        Kernel bandwidth in units of the **grid spacing**, so the absolute
        bandwidth is ``bandwidth_cells * (grid[1] - grid[0])``.  Default 0.5,
        the tengri convention (smooth across one neighbor on each side).

        Expressed in cells rather than in grid units on purpose: the window
        width is a shape, so it must be fixed from a static Python value.
        An absolute bandwidth would in practice be computed as
        ``0.5 * (grid[1] - grid[0])``, and inside a ``jit`` trace every
        ``jnp`` operation is staged into the jaxpr — so that expression is a
        tracer even when ``grid`` itself is a concrete constant, and cannot
        size anything.
    edges : array, shape (n + 1,) or None
        Precomputed bin edges from :func:`edges_for_grid`.  Passing them keeps
        the whole-axis edge construction out of the traced graph.

    Returns
    -------
    start : ndarray, shape (), int32
        Index of the first node in the window.  Clipped so the window stays in
        bounds, so ``start`` is not always ``nearest - half_width``.
    weights : ndarray, shape (2 * half_width + 1,)
        Non-negative weights summing to 1, equal element-for-element to
        ``compute_grid_weights(...)[start : start + weights.size]``, where
        ``half_width = ceil(3 * bandwidth_cells + 1/2)``.

    Raises
    ------
    ValueError
        If ``bandwidth_cells`` is not positive, or if ``grid`` is concrete and
        non-uniform or descending.

    Notes
    -----
    **JIT-compatible**: yes.
    **Gradient-safe**: yes.  ``start`` is integer-valued and carries no
    tangent, so ``d/dx`` flows through ``weights`` alone, exactly as in the
    dense path.  The window slides by one node when ``x`` crosses a cell
    midpoint; the node that leaves and the node that enters both carry zero
    weight at that point, so the interpolant stays continuous across the slide.

    The support bound is

    .. math::

        \\mathrm{half\\_width} = \\left\\lceil 3\\,b + \\frac{1}{2} \\right\\rceil

    where :math:`b` is ``bandwidth_cells`` (dimensionless).  The ``3`` is the
    triweight kernel's support in bandwidths and the ``1/2`` covers the offset
    of ``x`` from the nearest node, both in cells.  For the tengri convention
    :math:`b = 1/2` this gives ``half_width = 2``, a five-node window.

    References
    ----------
    .. [1] Hearin, A. P., Chaves-Montero, J., Becker, M. R., Alarcon, A. 2023,
       "DSPS: Differentiable stellar population synthesis", The Open Journal of
       Astrophysics, 6, 1. arXiv:2112.06830. DOI: 10.21105/astro.2112.06830

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.utils.interpolation import apply_grid_window, compute_grid_window
    >>> grid = jnp.linspace(0.0, 3.0, 250)
    >>> table = jnp.ones((250, 4))
    >>> start, w = compute_grid_window(1.234, grid)
    >>> int(w.shape[0])
    5
    >>> bool(jnp.allclose(apply_grid_window(table, start, w), 1.0))
    True
    """
    b = float(bandwidth_cells)
    if not b > 0.0:
        raise ValueError(f"bandwidth_cells must be positive; got {bandwidth_cells!r}")
    _check_uniform(grid)

    n = grid.shape[0]
    # Rounded before the ceil: the window size sets the returned shape and so
    # the compile-cache key, and 3*b + 0.5 lands on an integer only up to float
    # noise — without the snap an unlucky bandwidth silently widens the window.
    half_width = math.ceil(round(3.0 * b + 0.5, 9))
    width = min(2 * half_width + 1, n)

    if edges is None:
        edges = edges_for_grid(grid)

    # Traced-safe: dx and scatter are only ever used in arithmetic, never to
    # size anything.
    dx = grid[1] - grid[0]
    scatter = b * dx

    # Nearest node in O(1). Ties (x exactly on a cell midpoint) may break to
    # either neighbor; both are valid centers because the derivation only
    # assumes the offset from the chosen center is at most half a cell.
    nearest = jnp.clip(jnp.round((x - grid[0]) / dx).astype(jnp.int32), 0, n - 1)
    start = jnp.clip(nearest - half_width, 0, n - width)

    e = jax.lax.dynamic_slice_in_dim(edges, start, width + 1, axis=0)
    raw = tw_cuml_kern(x, e[:-1], scatter) - tw_cuml_kern(x, e[1:], scatter)
    total = jnp.sum(raw)
    # Same fallback as compute_grid_weights: if the kernel misses the grid
    # entirely, put all weight on the nearest node. Clipping above guarantees
    # that node is inside the window.
    fallback = jnp.zeros(width).at[nearest - start].set(1.0)
    return start, jnp.where(total > 0.0, raw / total, fallback)


def apply_grid_window(
    table: jnp.ndarray,
    start: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Contract a table's leading axis against a window from :func:`compute_grid_window`.

    Equivalent to ``jnp.tensordot(dense_weights, table, axes=([0], [0]))`` but
    reads only ``weights.size`` slices of ``table`` instead of all of them.
    Compute one window and apply it to every table sharing that axis.

    Parameters
    ----------
    table : array, shape (n, ...)
        Table whose leading axis lies on the interpolation grid.
    start : ndarray, shape (), int
        Window start index from :func:`compute_grid_window`.
    weights : array, shape (w,)
        Window weights from :func:`compute_grid_window`.

    Returns
    -------
    ndarray, shape (...)
        The interpolated table with the leading axis contracted away.

    Notes
    -----
    **JIT-compatible**: yes.  **Gradient-safe**: yes — differentiable in both
    ``table`` and (via ``weights``) the query point.
    """
    window = jax.lax.dynamic_slice_in_dim(table, start, weights.shape[0], axis=0)
    return jnp.tensordot(weights, window, axes=([0], [0]))
