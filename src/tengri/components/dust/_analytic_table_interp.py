# SPDX-License-Identifier: BSD-3-Clause
"""Cubic-spline lookup of log-carried band tables for the analytic dust precompute.

The band flux of a thermal-continuum dust model spans tens to hundreds of decades
over the declared temperature range (the Wien tail of a cold SED seen through a
mid-infrared filter), and is the log of a smooth function of the model parameters:
linear in the inverse temperature in the Wien limit, linear in ``beta`` through
``nu**beta``. The table therefore carries ``ln(flux)`` and interpolates it with a
tensor-product not-a-knot cubic spline in coordinates that make the log flux nearly
polynomial. Exponentiating the interpolant keeps the lookup positive and, because the
carrier is a log, representable in float32 where the linear flux underflows.

The spline is a linear operator on the node values. Its value at a query point is a
contraction of the table with one cardinal-weight vector per axis, so a lookup costs
one pass over the table, like the triweight kernel it replaces, and is C2 in every
query coordinate (a cubic spline is C2 across nodes), unlike the triweight smoother,
which averages neighboring nodes and is not node-exact.

Notes
-----
**JIT-compatible**: yes; the table, node coordinates and slope operators are static
arrays and the query enters through ``searchsorted`` indices and polynomial weights.
Queries are clipped to the node range, as :func:`tengri.utils.grid_interp.interp_nd_pchip`
does.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

__all__ = [
    "AXIS_TRANSFORMS",
    "LogSplineTable",
    "build_log_spline_table",
    "evaluate_log_spline",
    "not_a_knot_slope_operator",
    "slice_log_spline",
]

#: Coordinate in which each parameter axis is spline-interpolated: name ->
#: (forward, inverse). ``neg_inverse`` makes the Wien-tail log flux linear in the
#: temperature coordinate; ``log_shift_half`` resolves the steep dependence of the
#: Casey (2012) mid-infrared power-law mass on the slope near its lower bound.
AXIS_TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "identity": (lambda x: x, lambda y: y),
    "neg_inverse": (lambda x: -1.0 / x, lambda y: -1.0 / y),
    "log": (np.log, np.exp),
    "log_shift_half": (lambda x: np.log(x - 0.5), lambda y: np.exp(y) + 0.5),
}


def _jnp_transform(name: str) -> Callable:
    """The ``jnp`` spelling of a coordinate transform (queries are traced)."""
    return {
        "identity": lambda x: x,
        "neg_inverse": lambda x: -1.0 / x,
        "log": jnp.log,
        "log_shift_half": lambda x: jnp.log(x - 0.5),
    }[name]


def not_a_knot_slope_operator(x: np.ndarray) -> np.ndarray:
    """Matrix ``S`` with ``S @ y`` the node slopes of the not-a-knot cubic spline.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Strictly ascending node coordinates, ``n >= 1``.

    Returns
    -------
    ndarray, shape (n, n)
        The slope operator. For ``n = 1`` it is zero (a constant); for ``n = 2`` the spline is
        the chord, for ``n = 3`` the parabola through the three nodes, for ``n >= 4`` the
        not-a-knot cubic spline
        (third derivative continuous across the second and second-to-last nodes).

    Raises
    ------
    ValueError
        If ``x`` is empty or not strictly ascending.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 1 or np.any(np.diff(x) <= 0.0):
        raise ValueError(f"spline nodes must be strictly ascending, n >= 1; got {x!r}")
    if n == 1:
        return np.zeros((1, 1))
    h = np.diff(x)
    a = np.zeros((n, n))
    b = np.zeros((n, n))
    if n == 2:
        a[:] = np.eye(2)
        b[0] = (-1.0 / h[0], 1.0 / h[0])
        b[1] = b[0]
        return np.linalg.solve(a, b)
    if n == 3:
        # Parabola through the three nodes (Newton form): y'(x) = d0 + c (2 x - x0 - x1).
        d0 = np.array([-1.0 / h[0], 1.0 / h[0], 0.0])
        d1 = np.array([0.0, -1.0 / h[1], 1.0 / h[1]])
        c = (d1 - d0) / (x[2] - x[0])
        a[:] = np.eye(3)
        for i in range(3):
            b[i] = d0 + c * (2.0 * x[i] - x[0] - x[1])
        return b
    for i in range(1, n - 1):
        a[i, i - 1] = h[i]
        a[i, i] = 2.0 * (h[i - 1] + h[i])
        a[i, i + 1] = h[i - 1]
        b[i, i - 1] = -3.0 * h[i] / h[i - 1]
        b[i, i] = 3.0 * (h[i] / h[i - 1] - h[i - 1] / h[i])
        b[i, i + 1] = 3.0 * h[i - 1] / h[i]
    span0 = x[2] - x[0]
    a[0, 0] = h[1]
    a[0, 1] = span0
    b[0, 0] = -((h[0] + 2.0 * span0) * h[1] / h[0]) / span0
    b[0, 1] = ((h[0] + 2.0 * span0) * h[1] / h[0] - h[0] ** 2 / h[1]) / span0
    b[0, 2] = (h[0] ** 2 / h[1]) / span0
    span1 = x[-1] - x[-3]
    a[-1, -1] = h[-2]
    a[-1, -2] = span1
    b[-1, -1] = ((2.0 * span1 + h[-1]) * h[-2] / h[-1]) / span1
    b[-1, -2] = (h[-1] ** 2 / h[-2] - (2.0 * span1 + h[-1]) * h[-2] / h[-1]) / span1
    b[-1, -3] = -(h[-1] ** 2 / h[-2]) / span1
    return np.linalg.solve(a, b)


@dataclasses.dataclass(frozen=True)
class LogSplineTable:
    """A band table carried as ``ln(flux)`` with the data of its spline lookup.

    Attributes
    ----------
    log_phot : jnp.ndarray, shape (*axis_lengths, n_filters)
        Natural log of the band flux per unit absorbed luminosity.
    axes : tuple of ndarray
        Node values of each axis in physical units (ascending).
    transforms : tuple of str
        Key of :data:`AXIS_TRANSFORMS` used for each axis.
    coords : tuple of ndarray
        Node values in the transformed coordinate (ascending).
    slope_ops : tuple of ndarray
        :func:`not_a_knot_slope_operator` of each ``coords`` entry.
    """

    log_phot: jnp.ndarray
    axes: tuple[np.ndarray, ...]
    transforms: tuple[str, ...]
    coords: tuple[np.ndarray, ...]
    slope_ops: tuple[np.ndarray, ...]


def build_log_spline_table(
    log_phot: np.ndarray, axes: Sequence[np.ndarray], transforms: Sequence[str]
) -> LogSplineTable:
    """Assemble a :class:`LogSplineTable` from host ``ln(flux)`` values.

    Parameters
    ----------
    log_phot : ndarray, shape (*axis_lengths, n_filters)
        ``ln(flux)`` at the nodes, float64 (cast to the active JAX float on the way in).
    axes : sequence of ndarray
        Physical node values per axis, strictly ascending.
    transforms : sequence of str
        Keys of :data:`AXIS_TRANSFORMS`, one per axis.

    Returns
    -------
    LogSplineTable
    """
    axes_np = tuple(np.asarray(a, dtype=np.float64) for a in axes)
    coords = tuple(AXIS_TRANSFORMS[t][0](a) for a, t in zip(axes_np, transforms, strict=True))
    ops = tuple(not_a_knot_slope_operator(c) for c in coords)
    return LogSplineTable(
        log_phot=jnp.asarray(log_phot),
        axes=axes_np,
        transforms=tuple(transforms),
        coords=coords,
        slope_ops=ops,
    )


def _axis_weights(coord: jnp.ndarray, op: jnp.ndarray, xq) -> jnp.ndarray:
    """Cardinal-spline weights of the nodes at the transformed query ``xq``."""
    n = coord.shape[0]
    if n == 1:
        return jnp.ones((1,), dtype=coord.dtype)
    xq_c = jnp.clip(xq, coord[0], coord[-1])
    i = jnp.clip(jnp.searchsorted(coord, xq_c) - 1, 0, n - 2)
    h = coord[i + 1] - coord[i]
    t = (xq_c - coord[i]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    idx = jnp.arange(n)
    return h00 * (idx == i) + h01 * (idx == i + 1) + h * (h10 * op[i] + h11 * op[i + 1])


def evaluate_log_spline(table: LogSplineTable, point: Sequence) -> jnp.ndarray:
    """``ln(flux)`` of the table at ``point``, one scalar per axis.

    Parameters
    ----------
    table : LogSplineTable
    point : sequence of scalar
        Physical query value per axis, in axis order. Clipped to the node range.

    Returns
    -------
    jnp.ndarray, shape (n_filters,)
        Spline interpolant of ``ln(flux)``.
    """
    point = tuple(point)
    dtypes = [jnp.asarray(p).dtype for p in point]
    # A query made entirely of float32 values is evaluated in float32; anything else
    # in the dtype of the table.
    dtype = (
        jnp.float32 if dtypes and all(d == jnp.float32 for d in dtypes) else table.log_phot.dtype
    )
    out = table.log_phot.astype(dtype)
    for coord, op, name, p in zip(
        table.coords, table.slope_ops, table.transforms, point, strict=True
    ):
        w = _axis_weights(
            jnp.asarray(coord, dtype=dtype),
            jnp.asarray(op, dtype=dtype),
            _jnp_transform(name)(jnp.asarray(p, dtype=dtype)),
        )
        out = jnp.tensordot(w, out, axes=([0], [0]))
    return out


def slice_log_spline(table: LogSplineTable, fixed: Mapping[int, float]) -> LogSplineTable:
    """Collapse the axes in ``fixed`` by evaluating the spline at the given values.

    The collapsed table is the original interpolant restricted to the fixed
    coordinates, so its lookup equals the full lookup at those values.

    Parameters
    ----------
    table : LogSplineTable
    fixed : mapping of int to float
        Axis index -> physical value.

    Returns
    -------
    LogSplineTable
        Table over the surviving axes (the table itself if ``fixed`` is empty).
    """
    if not fixed:
        return table
    out = table.log_phot
    for axis in sorted(fixed, reverse=True):
        name = table.transforms[axis]
        xq = AXIS_TRANSFORMS[name][0](np.asarray(fixed[axis], dtype=np.float64))
        w = _axis_weights(
            jnp.asarray(table.coords[axis], dtype=out.dtype),
            jnp.asarray(table.slope_ops[axis], dtype=out.dtype),
            jnp.asarray(xq, dtype=out.dtype),
        )
        out = jnp.tensordot(w, out, axes=([0], [axis]))
    keep = [i for i in range(len(table.axes)) if i not in fixed]
    return LogSplineTable(
        log_phot=out,
        axes=tuple(table.axes[i] for i in keep),
        transforms=tuple(table.transforms[i] for i in keep),
        coords=tuple(table.coords[i] for i in keep),
        slope_ops=tuple(table.slope_ops[i] for i in keep),
    )
