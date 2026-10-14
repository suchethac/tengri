# SPDX-License-Identifier: BSD-3-Clause
"""Node axes that span what a model can reach, shared by the precompute adapters.

A precompute table holds the exact closure's value between its nodes and the edge value
beyond them, so a parameter whose prior reaches past the nodes is clipped and its gradient
is zero there. The rule implemented here, for one parameter:

* a ``Fixed(v)`` parameter reaches ``v``; a free parameter reaches its prior's finite bounds;
* a default axis spans the declared prior extended to include that reach, at the declared
  node density (the count scales with the span in the interpolation coordinate);
* a user-supplied axis must span the reach, or it is refused;
* a free prior with an infinite bound warns once and keeps the declared range.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from tengri.config.exceptions import GridSupportWarning

__all__ = [
    "active_support",
    "check_user_axis",
    "declared_bounds",
    "default_axis",
    "hull_axis",
]


def declared_bounds(declarations: Any, param_name: str) -> tuple[float, float]:
    """Declared prior range ``(lo, hi)`` of ``param_name`` in a component's declarations.

    Parameters
    ----------
    declarations : sequence of ParamDeclaration
        A component's ``PARAMS`` tuple; each entry carries a ``name`` and a ``prior``.
    param_name : str
        Parameter to look up.

    Returns
    -------
    tuple[float, float]
        ``prior.bounds`` of the declaration named ``param_name``.

    Raises
    ------
    KeyError
        If no declaration carries that name.
    """
    for decl in declarations:
        if decl.name == param_name:
            lo, hi = decl.prior.bounds
            return float(lo), float(hi)
    raise KeyError(f"{param_name} is not declared")


def active_support(
    param_name: str, parameters: Any, declared: tuple[float, float]
) -> tuple[float, float] | None:
    """Range of ``param_name`` the model can reach, or None when it is not bounded by the model.

    ``Fixed(v)`` gives ``(v, v)`` and a free parameter its prior's finite ``bounds``. ``None``
    means no model, a parameter the model does not declare, or a prior with an infinite bound; the
    last warns once, because the nodes then span the declared range and the lookup holds the edge
    value beyond it.

    Parameters
    ----------
    param_name : str
        Parameter name as declared on the component.
    parameters : Parameters or None
        The user's parameter specification.
    declared : tuple[float, float]
        The declared prior range of ``param_name``, quoted in the warning.

    Returns
    -------
    tuple[float, float] or None
        ``(lo, hi)`` the model reaches, or None.

    Warns
    -----
    GridSupportWarning
        When the parameter is free with an infinite prior bound.
    """
    if parameters is None:
        return None
    fixed = parameters.get_fixed_values()
    if param_name in fixed:
        return (fixed[param_name], fixed[param_name])
    if param_name not in parameters.free_params:
        return None
    lo, hi = parameters.get_distribution(param_name).bounds
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        return (float(lo), float(hi))
    warnings.warn(
        f"{param_name} has an unbounded prior; the nodes span its declared range "
        f"[{declared[0]:g}, {declared[1]:g}] and the lookup holds the edge value, with zero "
        f"gradient, beyond it. Give the prior finite bounds to widen the nodes.",
        GridSupportWarning,
        stacklevel=3,
    )
    return None


def default_axis(
    param_name: str,
    n_nodes: int,
    support: tuple[float, float] | None = None,
    *,
    declared: tuple[float, float],
    log_axis: bool = False,
) -> np.ndarray:
    """Node grid over the declared prior extended to ``support``, at the declared node density.

    Geometric for a logarithmic axis, linear otherwise. The count scales with the span in the
    interpolation coordinate (``ln`` on a logarithmic axis),
    ``ceil(n_nodes * span_axis / span_declared)``, never below ``n_nodes``, so the node spacing
    that the declared accuracy was measured at is kept when the support is wider. With
    ``support`` None, or inside the declared prior, the axis is the declared one exactly.

    Parameters
    ----------
    param_name : str
        Parameter name, quoted in errors.
    n_nodes : int
        Node count at the declared prior range.
    support : tuple[float, float] or None
        Reach from :func:`active_support`.
    declared : tuple[float, float]
        Declared prior range ``(lo, hi)``.
    log_axis : bool
        Geometric nodes and a ``ln`` interpolation coordinate when True.

    Returns
    -------
    ndarray, shape (n,)
        Node values, ``n >= n_nodes``.

    Raises
    ------
    ValueError
        If ``log_axis`` and the extended lower bound is not positive.
    """
    declared_lo, declared_hi = declared
    lo, hi = declared_lo, declared_hi
    if support is not None:
        lo, hi = min(lo, support[0]), max(hi, support[1])
    if log_axis and lo <= 0.0:
        raise ValueError(f"{param_name} reaches {lo:g}; a logarithmic node axis needs lo > 0.")
    coordinate = np.log if log_axis else np.asarray
    stretch = (coordinate(hi) - coordinate(lo)) / (
        coordinate(declared_hi) - coordinate(declared_lo)
    )
    n_axis = max(n_nodes, int(np.ceil(n_nodes * stretch)))
    if log_axis:
        return np.geomspace(lo, hi, n_axis, dtype=np.float64)
    return np.linspace(lo, hi, n_axis, dtype=np.float64)


def hull_axis(
    literal: np.ndarray, support: tuple[float, float] | None, *, log_axis: bool = False
) -> np.ndarray:
    """Literal default axis widened to include ``support``, at the literal's node density.

    The axis is never narrowed. When ``support`` lies inside the literal range, or is None, the
    literal is returned unchanged. Otherwise the axis spans the hull of the literal range and
    ``support``, with the node count scaled by the span in the interpolation coordinate
    (``ln`` on a logarithmic axis), ``ceil(n * span_hull / span_literal)``, never below ``n``.

    Parameters
    ----------
    literal : array_like, shape (n,)
        Default node values; the range they cover is the literal range.
    support : tuple[float, float] or None
        Reach from :func:`active_support`.
    log_axis : bool
        Geometric nodes and a ``ln`` interpolation coordinate when True.

    Returns
    -------
    ndarray, shape (m,)
        Node values, ``m >= n``.

    Raises
    ------
    ValueError
        If the literal spans no range, or ``log_axis`` and the extended lower bound is not
        positive.
    """
    literal = np.asarray(literal, dtype=np.float64)
    lit_lo, lit_hi = float(literal.min()), float(literal.max())
    if not lit_hi > lit_lo:
        raise ValueError("a default axis literal must span a non-zero range")
    lo, hi = (
        (lit_lo, lit_hi)
        if support is None
        else (
            min(lit_lo, support[0]),
            max(lit_hi, support[1]),
        )
    )
    if (lo, hi) == (lit_lo, lit_hi):
        return literal.copy()
    if log_axis and lo <= 0.0:
        raise ValueError(f"the axis reaches {lo:g}; a logarithmic node axis needs lo > 0.")
    coordinate = np.log if log_axis else np.asarray
    stretch = (coordinate(hi) - coordinate(lo)) / (coordinate(lit_hi) - coordinate(lit_lo))
    n_axis = max(literal.size, int(np.ceil(literal.size * stretch)))
    if log_axis:
        return np.geomspace(lo, hi, n_axis, dtype=np.float64)
    return np.linspace(lo, hi, n_axis, dtype=np.float64)


def check_user_axis(
    param_name: str, axis: np.ndarray, support: tuple[float, float] | None
) -> None:
    """Refuse a user-supplied axis that does not span the model's reach; warn below 4 nodes.

    Parameters
    ----------
    param_name : str
        Parameter name, quoted in messages.
    axis : ndarray, shape (n,)
        Supplied node values.
    support : tuple[float, float] or None
        Reach from :func:`active_support`; None skips the span check.

    Raises
    ------
    ValueError
        If the axis does not cover ``support``.

    Warns
    -----
    UserWarning
        If the axis has fewer than 4 nodes.
    """
    if support is not None and (axis.min() > support[0] or axis.max() < support[1]):
        raise ValueError(
            f"{param_name} nodes span [{axis.min():g}, {axis.max():g}] but the model reaches "
            f"[{support[0]:g}, {support[1]:g}]; the lookup would hold the edge value with zero "
            f"gradient beyond the nodes. Supply nodes covering the support."
        )
    if axis.size < 4:
        warnings.warn(
            f"{param_name} has {axis.size} nodes; the PCHIP lookup degrades to a parabola or a "
            f"chord below 4.",
            UserWarning,
            stacklevel=3,
        )
