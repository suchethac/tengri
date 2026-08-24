# SPDX-License-Identifier: BSD-3-Clause
"""Smooth grid interpolation via the triweight kernel (Hearin et al. 2023).

Provides C²-continuous weights for interpolating over 1-D grid axes.
Used by the CLOUDY nebular grid interpolation and available for any
future grid-based model component.

Note: the DSPS/SPS metallicity interpolation in ``dsps_wrapper.py``
keeps its own triweight kernel: that code mirrors upstream DSPS exactly
and should not be replaced.

References
----------

- Hearin et al. 2023, MNRAS, 521, 1741 (DSPS)

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
    index_space_interp: bool | None = None,
    on_out_of_grid: str = "clamp",
) -> jnp.ndarray:
    """Smooth triweight-kernel weights over a 1-D grid axis.

    Integrates the triweight kernel CDF between bin edges: the same approach
    as DSPS's ``triweighted_histogram``; and returns a weight vector that
    sums to 1.  Unlike piecewise-linear interpolation the weights transition
    smoothly through grid nodes, giving C²-continuous gradients.

    For non-uniform axes, automatically detects non-uniformity (max/min spacing
    ratio > 1 + 1e-6) and optionally maps the query coordinate to index space before
    computing weights. This ensures smooth interpolation across all intervals,
    not just those matching the first grid spacing.

    Parameters
    ----------
    x : float
        Query point on the axis.
    grid : array, shape (n,)
        Sorted grid node values (ascending). May be uniform or non-uniform.
    scatter : float
        Kernel bandwidth (same units as ``grid``).  Smaller values concentrate
        weight near the nearest node; larger values spread across more bins.
        Default 0.2, consistent with the DSPS lgmet_scatter convention.
        For non-uniform axes, this is interpreted in the physical coordinate space;
        the mapping to index space is automatic.
    edges : array, shape (n + 1,) or None
        Precomputed bin edges from :func:`edges_for_grid`.  When ``None``
        (default), edges are computed on the fly.
    index_space_interp : bool or None
        Whether to use index-space interpolation for non-uniform axes:
        - ``None`` (default): use pre-#1851 physical-space path, but emit a warning
          if non-uniformity is detected. This preserves backward compatibility while
          alerting callers to the #1851 degeneracy.
        - ``True``: apply index-space interpolation to detected non-uniform axes
          (see Notes below). Use this for Fritz, Nenkova, and other intentionally
          non-uniform grids.
        - ``False``: always use physical-space interpolation (legacy behavior).
    on_out_of_grid : str
        Behavior when the triweight kernel integrates to zero (query entirely
        outside grid bounds). Default ``"clamp"`` (clip to edge value, zero
        gradient outside). Alternative ``"nan"`` returns NaN out-of-grid for
        explicit signal. Issue #895: choosing "clamp" is the CIGALE convention
        (keeps gradient defined) but silently flattens the objective surface
        outside the grid. A prior wider than the grid will sit on a gradient-free
        plateau. Warnings at model construction flag this case so it can be
        addressed by narrowing the prior to the grid extent.

    Returns
    -------
    array, shape (n,)
        Non-negative weights summing to 1.

    Notes
    -----
    **Non-uniform axis handling (I6, #1851)**: When ``index_space_interp=True``
    and the grid spacing is non-uniform, the query coordinate is mapped to a
    piecewise-linear index-space coordinate via inverse linear interpolation, then
    triweight weights are computed in index space where spacing is uniform (=1).
    The interpolant is **C2 within intervals and C0 at nodes where adjacent
    spacings differ**; a property of piecewise-linear coordinate mapping (not a
    bug). The magnitude of the derivative jump at a node equals the ratio of the
    adjacent spacings. HMC/NUTS samplers and equivalently-robust optimizers tolerate
    such kinks; they are in the same smoothness class as linear interpolation.

    This eliminates the nearest-neighbor degeneracy (zero gradients over 67.5%
    of the range) that occurred on non-uniform axes when bandwidth was derived
    from the first interval alone. Uniform axes remain C² everywhere (value and
    gradient continuous).

    **Uniformity detection requires concrete (non-traced) axes.** The `grid` parameter
    must be a concrete array known at call time, not a traced JAX array. Template
    grids (SKIRTOR, dust libraries, etc.) are static and meet this requirement; this
    property holds for all static data accessed in ``compute_grid_weights`` calls.
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

    # Precompute bin edges (used by both paths)
    if edges is None:
        phys_edges = edges_for_grid(grid)
    else:
        phys_edges = jnp.asarray(edges, dtype=dt)

    # Detect non-uniformity: use dtype-aware tolerance to avoid false positives on
    # uniform float32 grids (e.g., linspace(0, 10, 50) in float32 has spacing ratio
    # ~1.00000465 due to accumulated rounding). Tolerance is 64 * machine epsilon
    # times the grid magnitude: this catches intentionally non-uniform grids
    # (e.g. geometric series with ratio 1.3+) while ignoring floating-point noise.
    # Guard against grids with < 2 elements: diff() creates an empty array, and
    # min/max would crash.
    n_spacings = n - 1

    # Guard: treat axes with <=1 nodes as uniform (no spacing reductions possible)
    if n_spacings <= 0:
        is_potentially_nonuniform = False
    else:
        # Compute non-uniformity detection unconditionally (no Python control flow).
        # This avoids TracerBoolConversionError when called inside jax.jit.
        spacings = jnp.diff(grid)
        min_spacing = jnp.min(spacings)
        max_spacing = jnp.max(spacings)
        # Dtype-aware tolerance: 64 eps * grid scale
        grid_scale = jnp.maximum(jnp.max(jnp.abs(grid)), 1.0)
        tolerance = 64.0 * jnp.finfo(grid.dtype).eps * grid_scale
        is_potentially_nonuniform = (max_spacing - min_spacing) > tolerance

    # Decide whether to use index-space interpolation based on the parameter.
    # The warning for default-None case must be emitted outside JIT (at call time)
    # and cannot be done in the traced code path.
    if index_space_interp is None:
        # Default: physical-space for backward compat, but warn if concrete and non-uniform
        # Try to detect if we're in a traced context (grid is a tracer)
        try:
            # If grid is concrete (not a tracer), check and warn
            grid_concrete = np.asarray(grid, dtype=float)
            if grid_concrete.ndim == 1 and grid_concrete.size >= 2:
                spacings_concrete = np.diff(grid_concrete)
                min_sp = np.min(spacings_concrete)
                max_sp = np.max(spacings_concrete)
                grid_scale_concrete = np.maximum(np.max(np.abs(grid_concrete)), 1.0)
                tol = 64.0 * np.finfo(grid_concrete.dtype).eps * grid_scale_concrete
                if (max_sp - min_sp) > tol:
                    # Non-uniform axis detected at call time. Emit warning.
                    #
                    # Through warn_measured, not warnings.warn: the message
                    # rounds the spacing ratio to two decimals for humans, and a
                    # consumer that has to regex the prose back out gets whatever
                    # that rounding left -- "1.00x" is anything in 0.995-1.005,
                    # which straddles "uniform" (#1645). The exact values ride on
                    # the instance instead. Enforced by
                    # tools/check_warning_payloads.py.
                    from tengri.config.exceptions import warn_measured

                    n_pts = len(grid_concrete)
                    axis_id = f"grid[{grid_concrete[0]:.4g}..{grid_concrete[-1]:.4g}] (n={n_pts})"
                    warn_measured(
                        f"Non-uniform axis {axis_id} detected (spacing ratio "
                        f"{max_sp / min_sp:.2f}x). #1851 degeneracy applies: "
                        "the triweight kernel will use physical-space interpolation, which "
                        "produces nearest-neighbor-like behavior over ~67.5% of the range. "
                        "Pass index_space_interp=True to opt-in to corrected index-space path, "
                        "or index_space_interp=False to silence this warning.",
                        stacklevel=3,
                        spacing_ratio=float(max_sp / min_sp),
                        min_spacing=float(min_sp),
                        max_spacing=float(max_sp),
                        n_nodes=int(n_pts),
                    )
        except (TypeError, ValueError):
            # grid is a tracer or otherwise not convertible to concrete array;
            # suppress warning to avoid repeated warnings in traced code.
            pass
        is_nonuniform = False
    elif index_space_interp:
        # Explicit opt-in: use index-space for detected non-uniform axes
        is_nonuniform = is_potentially_nonuniform
    else:
        # Explicit legacy mode: always physical-space, no warning
        is_nonuniform = False

    # Map query point to index space via piecewise-linear inverse interpolation.
    # idx_lower = floor(i) where i is the piecewise-linear index.
    idx_lower = jnp.clip(jnp.searchsorted(grid, x) - 1, 0, n - 2)
    x_lo = grid[idx_lower]
    x_hi = grid[idx_lower + 1]
    dx = x_hi - x_lo
    # Guard against zero spacing (shouldn't happen, but numerical safety)
    frac = jnp.where(dx > 0.0, (x - x_lo) / dx, 0.0)
    # Index-space coordinate: u = idx_lower + frac, lies in [0, n-1]
    x_index = idx_lower + frac

    # Edge array for the index-space grid (0, 1, 2, ..., n-1)
    index_grid = jnp.arange(n, dtype=dt)
    index_edges = edges_for_grid(index_grid)
    # Index-space bandwidth is always 0.5 (half a cell in index space)
    index_scatter = jnp.asarray(0.5, dtype=dt)

    # Validate on_out_of_grid mode at trace time (static check)
    if on_out_of_grid not in ("clamp", "nan"):
        raise ValueError(f"on_out_of_grid must be 'clamp' or 'nan', got {on_out_of_grid!r}")

    # Compute weights: use index space for non-uniform, physical space for uniform.
    # Both branches are traced but only one result is kept.
    # Handle on_out_of_grid choice: 'clamp' uses nearest-bin fallback, 'nan' returns NaN.
    # The choice is static (known at trace time) so we use plain Python if, not jnp.where.
    nan_fallback = jnp.full(n, jnp.nan) if on_out_of_grid == "nan" else None
    use_nan_fallback = on_out_of_grid == "nan"

    cuml_phys = tw_cuml_kern(x, phys_edges, scatter)
    raw_phys = cuml_phys[:-1] - cuml_phys[1:]
    total_phys = jnp.sum(raw_phys)
    nearest_phys = jnp.argmin(jnp.abs(grid - x))
    fallback_phys = nan_fallback if use_nan_fallback else jnp.zeros(n).at[nearest_phys].set(1.0)
    weights_phys = jnp.where(total_phys > 0.0, raw_phys / total_phys, fallback_phys)

    cuml_idx = tw_cuml_kern(x_index, index_edges, index_scatter)
    raw_idx = cuml_idx[:-1] - cuml_idx[1:]
    total_idx = jnp.sum(raw_idx)
    nearest_idx = jnp.argmin(jnp.abs(index_grid - x_index))
    fallback_idx = nan_fallback if use_nan_fallback else jnp.zeros(n).at[nearest_idx].set(1.0)
    weights_idx = jnp.where(total_idx > 0.0, raw_idx / total_idx, fallback_idx)

    # Select the appropriate result based on uniformity
    return jnp.where(is_nonuniform, weights_idx, weights_phys)


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
        g0 = np.asarray(grid)
    except Exception:  # a JAX tracer cannot be converted, nothing to check
        return
    if not np.issubdtype(g0.dtype, np.floating):
        return
    g = g0.astype(float)
    if g.ndim != 1 or g.size < 2:
        raise ValueError(f"grid must be 1-D with at least 2 nodes; got shape {g.shape}")
    d = np.diff(g)
    if d[0] <= 0.0 or not np.allclose(d, d[0], rtol=_uniform_rtol(g0), atol=0.0):
        raise ValueError(
            "compute_grid_window requires a uniform ascending grid; got spacings in "
            f"[{d.min():.6g}, {d.max():.6g}] (tolerance {_uniform_rtol(g0):.3g} for "
            f"{g0.dtype}). Use compute_grid_weights for a non-uniform axis."
        )


def _uniform_rtol(grid: np.ndarray) -> float:
    r"""Uniformity tolerance for *grid*, derived from its dtype rather than written.

    Parameters
    ----------
    grid : ndarray, shape (n,)
        Grid nodes **in their original dtype**; the tolerance depends on the
        precision the grid was built in, so this must be called before any
        upcast to float64.

    Returns
    -------
    float
        Relative tolerance for comparing adjacent spacings [dimensionless].

    Notes
    -----
    A ``linspace`` is exactly uniform in exact arithmetic; stored in a finite
    dtype its nodes carry absolute error up to :math:`\epsilon \max|g|`, so
    adjacent differences carry up to twice that, and *relative* to the spacing
    :math:`\Delta` the spread is

    .. math::

        \frac{\delta(\Delta)}{\Delta} \;\lesssim\; \frac{\epsilon \max|g|}{\Delta}

    with :math:`\epsilon` the dtype's machine epsilon (dimensionless),
    :math:`\max|g|` the largest node magnitude and :math:`\Delta` the nominal
    spacing (same units as the grid). The factor 8 is headroom over that bound.

    The former hardcoded ``rtol=1e-9`` was a float64 tolerance. Measured on the
    z-grid ``linspace(0, 4, 494)``: float64 spreads by 5.5e-14 and passes, while
    **float32 spreads by 2.9e-05 and fails**; so a genuinely uniform grid was
    rejected in float32 and `compute_grid_window` raised on the default fit path
    (``Fitter`` resolves ``approx="auto"`` to ``WavePrecomp``, whose z-table
    reaches this guard). Same class as the float64 literals in
    :func:`tengri.utils.scale.representable_floor` and its siblings: a numeric
    constant that encodes an assumption about the number system, written rather
    than derived (#1206).

    **float64 is unchanged by construction**: the derived bound there is ~1e-13,
    below the 1e-9 floor this keeps, so float64 grids are judged by exactly the
    old tolerance.
    """
    eps = float(np.finfo(grid.dtype).eps)
    g = np.asarray(grid, dtype=float)
    d0 = abs(float(g[1] - g[0]))
    if d0 == 0.0:
        return 1e-9
    scale = float(np.max(np.abs(g)))
    return max(1e-9, 8.0 * eps * scale / d0)


def compute_grid_window(
    x: float,
    grid: jnp.ndarray,
    bandwidth_cells: float = 0.5,
    edges: jnp.ndarray | None = None,
    on_out_of_grid: str = "clamp",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kernel-supported slice of :func:`compute_grid_weights` on a uniform grid.

    :func:`compute_grid_weights` returns a dense weight vector over the whole
    axis, but the triweight kernel has support only on ``|x - m| < 3 * scatter``.
    On a uniform grid that is a fixed number of adjacent nodes regardless of how
    fine the grid is, so the dense vector is mostly exact zeros.  This function
    returns just the supported window; a start index and the weights at
    ``grid[start: start + 2 * half_width + 1]`` : for use with
    :func:`apply_grid_window`.

    The dropped weights are **exactly** zero (:func:`tw_cuml_kern` clamps the
    CDF outside ``|z| > 3``), so this is an algebraic identity, not an
    approximation: the interpolated value, its gradient, and its C²-continuity
    are all unchanged.  What changes is cost; contracting a ``(n, ...)`` table
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
        ``jnp`` operation is staged into the jaxpr; so that expression is a
        tracer even when ``grid`` itself is a concrete constant, and cannot
        size anything.
    edges : array, shape (n + 1,) or None
        Precomputed bin edges from :func:`edges_for_grid`.  Passing them keeps
        the whole-axis edge construction out of the traced graph.
    on_out_of_grid : str
        Behavior when the triweight kernel integrates to zero (query entirely
        outside grid bounds). Default ``"clamp"`` (clip to edge value, zero
        gradient outside). Alternative ``"nan"`` returns NaN out-of-grid.
        See :func:`compute_grid_weights` for details. Issue #895.

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
    .. [1] Hearin, A. P., Chaves-Montero, J., Alarcon, A., Becker, M. R.,
       Benson, A. 2023, "DSPS: Differentiable stellar population synthesis",
       MNRAS, 521, 1741. arXiv:2112.06830. DOI: 10.1093/mnras/stad456

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
    # Validate on_out_of_grid mode at trace time (static check)
    if on_out_of_grid not in ("clamp", "nan"):
        raise ValueError(f"on_out_of_grid must be 'clamp' or 'nan', got {on_out_of_grid!r}")

    b = float(bandwidth_cells)
    if not b > 0.0:
        raise ValueError(f"bandwidth_cells must be positive; got {bandwidth_cells!r}")
    _check_uniform(grid)

    n = grid.shape[0]
    # Rounded before the ceil: the window size sets the returned shape and so
    # the compile-cache key, and 3*b + 0.5 lands on an integer only up to float
    # noise: without the snap an unlucky bandwidth silently widens the window.
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
    # Use plain Python if (not jnp.where) since on_out_of_grid is static.
    use_nan_fallback = on_out_of_grid == "nan"
    nan_fallback = jnp.full(width, jnp.nan) if use_nan_fallback else None
    fallback = nan_fallback if use_nan_fallback else jnp.zeros(width).at[nearest - start].set(1.0)
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
    **JIT-compatible**: yes.  **Gradient-safe**: yes, differentiable in both
    ``table`` and (via ``weights``) the query point.
    """
    window = jax.lax.dynamic_slice_in_dim(table, start, weights.shape[0], axis=0)
    return jnp.tensordot(weights, window, axes=([0], [0]))
