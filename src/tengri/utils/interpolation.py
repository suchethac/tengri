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

import jax
import jax.numpy as jnp
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
