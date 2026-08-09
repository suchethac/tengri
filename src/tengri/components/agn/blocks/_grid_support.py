# SPDX-License-Identifier: BSD-3-Clause
"""Grid support: the second, implicit support a template-backed block carries.

A :class:`~tengri.protocols.component.ParamDeclaration` records one support —
its prior. A block that interpolates a template library carries another: the
extent of the grid axes it interpolates over. Values outside those axes are
clipped onto the edge node, so the SED is **bit-identical** across the excess
and the gradient there is **exactly zero** (``jnp.clip`` is flat outside its
bounds). Nothing raises, nothing warns, and no NaN appears — the parameter
simply stops doing anything (#1586).

Why this cannot live on the declaration
---------------------------------------
``agn_log_mbh`` / ``agn_log_ledd`` are shared with the analytic disc models
(``kd18_disc_model``, ``adaf``, ``unified``, ``disc``), which have no grid and
legitimately want the wide physical support. Narrowing the shared declaration
would wrongly constrain them, and a
:attr:`~tengri.protocols.component.ParamDeclaration.bound_check` is global to
the declaration, so it cannot express "only when this block is selected".
The constraint is a property of the ``(block, parameter)`` pair, so it is
recorded here and checked by
:func:`~tengri.components.agn.blocks.runner.validate_block_recipe`.

Contrast with ``agn_tau``, whose declaration *can* carry its grid extent
("must be within the CLUMPY grid extent [5, 150]") precisely because no
grid-free model consumes it.

Adding a block
--------------
Register a zero-argument callable returning ``{param: (lo, hi)}`` read from
the grid file itself. Deriving the bounds from the data keeps them from going
stale if the packaged grid is ever rebuilt; a hand-written literal would not.
"""

from __future__ import annotations

import math
from collections.abc import Callable

#: A block's grid support: ``() -> {param_name: (lo, hi)}``.
GridSupportFn = Callable[[], dict[str, tuple[float, float]]]


def _slone_netzer_support() -> dict[str, tuple[float, float]]:
    """Read the SN12 disc grid axes (imported lazily — h5py + file I/O)."""
    from tengri.components.agn.slone_netzer import slone_netzer_grid_support

    return slone_netzer_grid_support()


#: ``(category, block name)`` -> accessor for that block's grid support.
#:
#: Only template-backed blocks appear. A block absent from this table is
#: unconstrained, which is the correct default for every closed-form model.
AGN_BLOCK_GRID_SUPPORT: dict[tuple[str, str], GridSupportFn] = {
    ("disc", "slone_netzer"): _slone_netzer_support,
}


def block_grid_support(category: str, name: str) -> dict[str, tuple[float, float]]:
    """Return the grid support of one block, or ``{}`` if it has none.

    Parameters
    ----------
    category : str
        Pipeline stage, e.g. ``'disc'``.
    name : str
        Block name, e.g. ``'slone_netzer'``.

    Returns
    -------
    support : dict[str, tuple[float, float]]
        ``{param_name: (lo, hi)}``. Empty when the block is not
        template-backed **or** when its grid is not installed — an absent
        data file must not break model construction, since the block itself
        raises a clear :class:`FileNotFoundError` if it is ever called.

    Notes
    -----
    **JIT-compatible**: not applicable — composition-time only.
    """
    accessor = AGN_BLOCK_GRID_SUPPORT.get((category, name))
    if accessor is None:
        return {}
    try:
        return accessor()
    except FileNotFoundError:
        # Grid not installed. Nothing to compare against, and the block's own
        # loader already raises an actionable error at call time. Narrow on
        # purpose: any other exception is a real defect and must propagate.
        return {}


#: Slack on the containment test, relative to the grid's own width.
#:
#: A prior written to match a grid axis is normally transcribed to a handful of
#: decimals, so it can overhang the true bound by a few ulp. Comparing exactly
#: reports that transcription as a defect and then prints a self-contradictory
#: "0% of its range lies outside". A sliver this thin is not reachable by any
#: fit, so treat it as contained (CLAUDE.md: compare floats with a tolerance,
#: never ``==``).
_CONTAINMENT_RTOL = 1e-6


def is_contained(active: tuple[float, float], grid: tuple[float, float]) -> bool:
    """Whether an active support fits inside a grid's support.

    Parameters
    ----------
    active : tuple[float, float]
        ``(lo, hi)`` the parameter can actually take.
    grid : tuple[float, float]
        ``(lo, hi)`` covered by the template grid.

    Returns
    -------
    contained : bool
        ``True`` when no reachable value can be clipped, within
        :data:`_CONTAINMENT_RTOL` of the grid width.

    Notes
    -----
    **JIT-compatible**: not applicable — composition-time only.
    """
    a_lo, a_hi = active
    g_lo, g_hi = grid
    width = g_hi - g_lo
    tol = _CONTAINMENT_RTOL * (width if math.isfinite(width) and width > 0.0 else 1.0)
    return (g_lo - tol) <= a_lo and a_hi <= (g_hi + tol)


def live_fraction(active: tuple[float, float], grid: tuple[float, float]) -> float:
    """Fraction of an active support that lies inside a grid's support.

    Parameters
    ----------
    active : tuple[float, float]
        ``(lo, hi)`` the parameter can actually take — a prior's bounds, or
        ``(v, v)`` for a fixed value.
    grid : tuple[float, float]
        ``(lo, hi)`` covered by the template grid.

    Returns
    -------
    fraction : float
        In ``[0, 1]``. ``1.0`` means fully contained (no clipping possible);
        ``0.0`` means every reachable value clips onto an edge node, so the
        parameter is entirely inert. A zero-width ``active`` (a fixed value)
        yields ``1.0`` or ``0.0``. An unbounded ``active`` yields ``0.0``,
        since no finite grid can contain it.

    Notes
    -----
    **JIT-compatible**: not applicable — composition-time only.
    """
    a_lo, a_hi = active
    g_lo, g_hi = grid
    width = a_hi - a_lo
    if not math.isfinite(width):
        return 0.0
    if width <= 0.0:
        return 1.0 if g_lo <= a_lo <= g_hi else 0.0
    overlap = min(a_hi, g_hi) - max(a_lo, g_lo)
    return max(0.0, min(1.0, overlap / width))


__all__ = [
    "AGN_BLOCK_GRID_SUPPORT",
    "GridSupportFn",
    "block_grid_support",
    "is_contained",
    "live_fraction",
]
