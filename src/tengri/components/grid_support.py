# SPDX-License-Identifier: BSD-3-Clause
"""Grid support: the second, implicit support a template-backed component carries.

A :class:`~tengri.protocols.component.ParamDeclaration` records one support --
its prior. A component that interpolates a template library carries another:
the extent of the grid axes it interpolates over. Values outside those axes are
clipped onto the edge node, so the SED is **bit-identical** across the excess
and the gradient there is **exactly zero** (``jnp.clip`` is flat outside its
bounds). Nothing raises, nothing warns, and no NaN appears -- the parameter
simply stops doing anything (#1586).

Why this cannot live on the declaration
---------------------------------------
The parameters concerned are shared. ``agn_log_mbh`` / ``agn_log_ledd`` are
consumed by the analytic disc models (``kd18_disc_model``, ``adaf``,
``unified``, ``disc``), which have no grid and legitimately want the wide
physical support. ``dust_lgU`` is consumed by both ``astrodust`` and
``draine2021_pah_ir``, whose grids need not agree. A
:attr:`~tengri.protocols.component.ParamDeclaration.bound_check` is global to
the declaration, so it cannot express "only when this component is selected".

The constraint is therefore a property of the ``(component, parameter)`` pair
and is recorded here, keyed by the same selector names a user writes in the
:meth:`~tengri.forward.sed_model.SEDModel.build` grammar.

Contrast with ``agn_tau``, whose declaration *can* carry its grid extent
("must be within the CLUMPY grid extent [5, 150]") precisely because no
grid-free model consumes it.

Registering a component
-----------------------
Add a zero-argument callable returning ``{param: (lo, hi)}`` read from the grid
file itself, keyed by ``(selector, name)`` -- e.g. ``("dust.emission",
"themis")``. Deriving the bounds from the data keeps them from going stale if a
packaged grid is rebuilt; a hand-written literal would not.

**Return the axes the model actually interpolates on, not the raw file
contents.** Several loaders transform an axis (``create_themis_from_grid``
rescales ``qhac`` from the FSPS convention to CIGALE's), and an accessor that
reads the raw dataset would report a support the model never sees.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

#: A component's grid support: ``() -> {param_name: (lo, hi)}``.
GridSupportFn = Callable[[], dict[str, tuple[float, float]]]


def _slone_netzer_support() -> dict[str, tuple[float, float]]:
    """Read the SN12 disc grid axes (imported lazily -- h5py + file I/O)."""
    from tengri.components.agn.slone_netzer import slone_netzer_grid_support

    return slone_netzer_grid_support()


def _dust_emission_support(name: str) -> GridSupportFn:
    """Build an accessor for one template-backed dust emission model."""

    def _accessor() -> dict[str, tuple[float, float]]:
        from tengri.components.dust.emission_templates import dust_emission_grid_support

        return dust_emission_grid_support(name)

    return _accessor


#: ``(selector, component name)`` -> accessor for that component's grid support.
#:
#: ``selector`` is the dotted path used by the build grammar (``"agn.disc"``,
#: ``"dust.emission"``). Only template-backed components appear; a component
#: absent from this table is unconstrained, which is the correct default for
#: every closed-form model.
GRID_SUPPORT: dict[tuple[str, str], GridSupportFn] = {
    ("agn.disc", "slone_netzer"): _slone_netzer_support,
    **{
        ("dust.emission", _name): _dust_emission_support(_name)
        # Every selectable spelling, aliases included: the menu exposes
        # 'draine_li2007' and 'dl07_tabulated' alongside 'dl07', and a census
        # that covers only the canonical name leaves the others unchecked.
        # tests/regression/dust/test_issue_1586_dust_grid_support.py fails if a
        # template-backed menu entry is missing from this table.
        for _name in (
            "themis",
            "dl07",
            "dl07_tabulated",
            "draine_li2007",
            "dl14",
            "draine_li2014",
            "dale2014",
            "dale2014_cigale",
            "schreiber2016",
            "schreiber2018",
            "bosa",
            "astrodust",
        )
    },
}
# dh02_ce01 is deliberately absent: its only grid axis is L_TIR, derived from
# L_absorbed by energy balance rather than set by the user, so no prior can
# overhang it. See _DUST_EMISSION_GRID_AXES for the same reasoning on bosa.


def grid_support(selector: str, name: str) -> dict[str, tuple[float, float]]:
    """Return the grid support of one component, or ``{}`` if it has none.

    Parameters
    ----------
    selector: str
        Dotted selector path, e.g. ``'agn.disc'`` or ``'dust.emission'``.
    name: str
        Component name, e.g. ``'slone_netzer'`` or ``'themis'``.

    Returns
    -------
    support: dict[str, tuple[float, float]]
        ``{param_name: (lo, hi)}``. Empty when the component is not
        template-backed **or** when its grid is not installed -- an absent
        data file must not break model construction, since the component
        itself raises a clear :class:`FileNotFoundError` if it is ever called.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only.
    """
    accessor = GRID_SUPPORT.get((selector, name))
    if accessor is None:
        return {}
    try:
        return accessor()
    except FileNotFoundError:
        # Grid not installed. Nothing to compare against, and the component's
        # own loader already raises an actionable error at call time. Narrow on
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
    active: tuple[float, float]
        ``(lo, hi)`` the parameter can actually take.
    grid: tuple[float, float]
        ``(lo, hi)`` covered by the template grid.

    Returns
    -------
    contained: bool
        ``True`` when no reachable value can be clipped, within
        :data:`_CONTAINMENT_RTOL` of the grid width.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only.
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
    active: tuple[float, float]
        ``(lo, hi)`` the parameter can actually take -- a prior's bounds, or
        ``(v, v)`` for a fixed value.
    grid: tuple[float, float]
        ``(lo, hi)`` covered by the template grid.

    Returns
    -------
    fraction: float
        In ``[0, 1]``. ``1.0`` means fully contained (no clipping possible);
        ``0.0`` means every reachable value clips onto an edge node, so the
        parameter is entirely inert. A zero-width ``active`` (a fixed value)
        yields ``1.0`` or ``0.0``. An unbounded ``active`` yields ``0.0``,
        since no finite grid can contain it -- callers must distinguish that
        case before quoting a percentage, which :func:`describe_clipping` does.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only.
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


def describe_clipping(active: tuple[float, float], grid: tuple[float, float]) -> str | None:
    """Describe how an active support overhangs a grid, or ``None`` if it fits.

    Returns only the component-agnostic clause, so each caller keeps its own
    framing and existing message wording stays byte-identical.

    Parameters
    ----------
    active: tuple[float, float]
        ``(lo, hi)`` the parameter can actually take.
    grid: tuple[float, float]
        ``(lo, hi)`` covered by the template grid.

    Returns
    -------
    detail: str or None
        ``None`` when contained. Otherwise a clause such as ``"40% of its
        range [6, 10] lies outside the grid extent [7.4, 9.8] and is silently
        clipped onto an edge node"``.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only.
    """
    if is_contained(active, grid):
        return None
    a_lo, a_hi = active
    g_lo, g_hi = grid
    extent = f"[{g_lo:g}, {g_hi:g}]"
    if a_lo == a_hi:
        return (
            f"the fixed value {a_lo:g} lies outside the grid extent "
            f"{extent}, so it is clipped onto the nearest edge node"
        )
    if not (math.isfinite(a_lo) and math.isfinite(a_hi)):
        # An unbounded prior (e.g. an untruncated Gaussian) is NOT inert --
        # most of its mass may sit on the grid. Only the tails clip, so say
        # that and do not quote a percentage: the fraction of an infinite
        # support is not informative.
        return (
            f"its support [{a_lo:g}, {a_hi:g}] is unbounded, so the tails "
            f"beyond the grid extent {extent} are clipped onto an edge node"
        )
    live = live_fraction(active, grid)
    if live == 0.0:
        return (
            f"its whole range [{a_lo:g}, {a_hi:g}] lies outside the grid "
            f"extent {extent}, so the parameter is entirely inert -- every "
            "value gives the same SED"
        )
    return (
        f"{100.0 * (1.0 - live):.0f}% of its range [{a_lo:g}, {a_hi:g}] lies "
        f"outside the grid extent {extent} and is silently clipped onto an "
        "edge node"
    )


def check_grid_support(
    selected: Iterable[tuple[str, str]],
    param_support: Mapping[str, tuple[float, float]],
) -> list[tuple[str, str, str, str, tuple[float, float]]]:
    """Find every selected component whose grid cannot cover an active support.

    Parameters
    ----------
    selected: iterable of (str, str)
        ``(selector, name)`` pairs for the components in play, e.g.
        ``[("dust.emission", "themis")]``.
    param_support: mapping of str to (float, float)
        ``{param_name: (lo, hi)}`` the range each parameter can actually take
        -- a prior's bounds, or ``(v, v)`` for a fixed value.

    Returns
    -------
    findings: list of tuple
        ``(selector, name, param_name, detail, grid_extent)``, one per
        offending ``(component, parameter)`` pair. Empty when everything fits.

    Notes
    -----
    **JIT-compatible**: not applicable -- composition-time only.
    """
    findings: list[tuple[str, str, str, str, tuple[float, float]]] = []
    if not param_support:
        return findings
    for selector, name in selected:
        for pname, extent in grid_support(selector, name).items():
            active = param_support.get(pname)
            if active is None:
                continue
            detail = describe_clipping(active, extent)
            if detail is not None:
                findings.append((selector, name, pname, detail, extent))
    return findings


__all__ = [
    "GRID_SUPPORT",
    "GridSupportFn",
    "check_grid_support",
    "describe_clipping",
    "grid_support",
    "is_contained",
    "live_fraction",
]
