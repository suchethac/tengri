# SPDX-License-Identifier: BSD-3-Clause
"""Grid support: the second, implicit support a template-backed component carries.

A :class:`~tengri.protocols.component.ParamDeclaration` records one support --
its prior. A component that interpolates a template library carries another:
the extent of the grid axes it interpolates over. Values outside those axes are
clipped onto the edge node, so the SED is **bit-identical** across the excess
and the gradient there is **exactly zero** (``jnp.clip`` is flat outside its
bounds). Nothing raises, nothing warns, and no NaN appears -- the parameter
simply stops doing anything (#1586).

One registered entry, ``("neb", "cue")``, is not a grid at all but a neural
network's trained footprint (#2569): nothing clips there either, so its
support narrows and warns for a different reason -- see
:data:`EXTRAPOLATING_SUPPORT`.

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


def _kd18_agnfitter_support() -> dict[str, tuple[float, float]]:
    """Read the KD18-agnfitter disc grid axes (imported lazily -- h5py + file I/O)."""
    from tengri.components.agn.kd18_agnfitter import kd18_agnfitter_grid_support

    return kd18_agnfitter_grid_support()


def _kd18_agnfitter_warmindex_support() -> dict[str, tuple[float, float]]:
    """Read the KD18-agnfitter-warmindex disc grid axes (imported lazily)."""
    from tengri.components.agn.kd18_agnfitter import kd18_agnfitter_warmindex_grid_support

    return kd18_agnfitter_warmindex_grid_support()


def _dust_emission_support(name: str) -> GridSupportFn:
    """Build an accessor for one template-backed dust emission model."""

    def _accessor() -> dict[str, tuple[float, float]]:
        from tengri.components.dust.emission_templates import dust_emission_grid_support

        return dust_emission_grid_support(name)

    return _accessor


def _cloudy_grid_support() -> dict[str, tuple[float, float]]:
    """Read the default Cloudy grid's log U and log met axes (#2460).

    Resolves the grid the same way the backend itself does --
    ``Parameters._default_cloudy_grid`` in ``tengri.parameters.parameters``,
    which honors ``$TENGRI_DATA_DIR`` -- rather than a hardcoded path, so a
    worktree without a repo-level ``data/`` still finds the grid a real build
    would use. This accessor has no SSP context (composition-time, before any
    per-build isochrone match runs -- see #2426), so it reads the isochrone
    that resolves when nothing else is known (``cloudy_grid_mist.h5`` if
    present). ``neb_logU`` is exact for every shipped isochrone variant: the
    log U axis (``[-4, -1]``, 7 nodes) is identical across
    ``cloudy_grid_{mist,prsc,pdva,bpss}.h5``, verified by direct inspection.
    ``neb_logZ_gas`` is NOT: ``cloudy_grid_bpss.h5``'s ``log_met`` axis spans
    ``[-1.3, 0.3]`` where the others span ``[-1.98, 0.2]``, so the registered
    bound is exact only when the build resolves to the same isochrone this
    accessor found. That is the same imprecision already tolerated for
    ``agn_log_ledd`` (see ``groups.py``'s ``_narrow_free_priors_to_grid``): a
    narrowing that undershoots or overshoots by a modeling-relevant amount is
    a decision for the per-build warning to report, not a reason to withhold
    the registration for the common (matching) case.

    Unit of ``log_met``: the file attrs mislabel it "absolute metallicity",
    but the values themselves (11 nodes spanning roughly -2.0 to 0.2, the
    classic FSPS/Byler et al. 2017 metallicity grid) cannot be absolute
    log10(Z) -- that would put Z up to 10**0.2 ~ 1.6, past unity. They are
    log10(Z/Zsun), confirmed by ``components/nebular/cloudy_grid.py``'s own
    loader, which adds ``_LOG10_ZSUN`` to this same raw axis to form its
    internal *absolute* ``line_log_met``/``cont_log_met``. The declared
    ``neb_logZ_gas`` prior is likewise log10(Z/Zsun) (``_params.py``), so the
    raw file axis is registered as-is, with no offset.
    """
    import h5py

    from tengri.parameters.parameters import Parameters

    grid_path = Parameters._default_cloudy_grid()
    if grid_path is None:
        raise FileNotFoundError("No Cloudy grid found (see $TENGRI_DATA_DIR)")

    with h5py.File(grid_path, "r") as f:
        if "continuum/axes" not in f:
            raise FileNotFoundError(f"Grid {grid_path} missing continuum/axes")

        log_U = f["continuum/axes/log_U"][:]
        log_met = f["continuum/axes/log_met"][:]

        return {
            "neb_logU": (float(log_U.min()), float(log_U.max())),
            "neb_logZ_gas": (float(log_met.min()), float(log_met.max())),
        }


def _cb19_grid_support() -> dict[str, tuple[float, float]]:
    """Read CB_19's default template grid's log U axis (#2460).

    ``CB19Backend`` clips ``neb_logU`` onto its grid axis exactly like
    ``CloudyGridBackend`` (``cloudy_cb19.py``'s ``_frac_idx``:
    ``jnp.clip(val, grid[0], grid[-1])``), so it shares the #1586 mechanism
    and belongs in this registry. Only ``neb_logU`` is registered: the grid's
    metallicity axis is ``log_OH_total`` (12 + log10(O/H)), not
    log10(Z/Zsun), and converting between the two abundance scales exactly
    is out of scope here -- ``neb_logZ_gas`` keeps its declared support on
    this backend.
    """
    import h5py

    from tengri._data_setup import package_or_env_data_path

    grid_path = package_or_env_data_path("cb19_templates.h5")
    if not grid_path.exists():
        raise FileNotFoundError(f"CB_19 grid not found: {grid_path}")

    with h5py.File(grid_path, "r") as f:
        if "axes/log_U" not in f:
            raise FileNotFoundError(f"Grid {grid_path} missing axes/log_U")
        log_U = f["axes/log_U"][:]

    return {"neb_logU": (float(log_U.min()), float(log_U.max()))}


def _mappings_stellar_grid_support() -> dict[str, tuple[float, float]]:
    """Read MAPPINGS V stellar backend's default grid's log U axis (#2460).

    ``MappingsPhotoStellarBackend`` interpolates ``neb_logU`` via
    ``_interp_index_weight`` (``mappings_photo.py``), which clips to the
    grid axis the same way ``CloudyGridBackend`` does. Reads the ``sb99``/
    ``cpr`` subgroup axis, the backend's own constructor defaults
    (``model="sb99"``, ``density="cpr"``); a build using ``model="bpass"``
    or ``density="cdn"`` is narrowed to this axis too, since every shipped
    MAPPINGS V variant shares the same ``logU_axis`` (Flury et al. 2024's
    grid spans one ionization-parameter range regardless of stellar model
    or density structure).
    """
    import h5py

    from tengri._data_setup import package_or_env_data_path

    grid_path = package_or_env_data_path("flury2024_grids.h5")
    if not grid_path.exists():
        raise FileNotFoundError(f"MAPPINGS V grid not found: {grid_path}")

    with h5py.File(grid_path, "r") as f:
        group_key = "sb99/cpr/logU_axis"
        if group_key not in f:
            raise FileNotFoundError(f"Grid {grid_path} missing {group_key}")
        log_U = f[group_key][:]

    return {"neb_logU": (float(log_U.min()), float(log_U.max()))}


def _mappings_agn_grid_support() -> dict[str, tuple[float, float]]:
    """Read MAPPINGS V AGN backend's default grid's log U axis (#2460).

    Same mechanism and grid file as :func:`_mappings_stellar_grid_support`,
    under the ``agn_oxaf/<density>`` subgroup instead of ``<model>/<density>``
    (``MappingsPhotoAGNBackend`` has no stellar-model axis). Reads the
    backend's own default ``density="cpr"``.
    """
    import h5py

    from tengri._data_setup import package_or_env_data_path

    grid_path = package_or_env_data_path("flury2024_grids.h5")
    if not grid_path.exists():
        raise FileNotFoundError(f"MAPPINGS V grid not found: {grid_path}")

    with h5py.File(grid_path, "r") as f:
        group_key = "agn_oxaf/cpr/logU_axis"
        if group_key not in f:
            raise FileNotFoundError(f"Grid {grid_path} missing {group_key}")
        log_U = f[group_key][:]

    return {"neb_logU": (float(log_U.min()), float(log_U.max()))}


def _cue_grid_support() -> dict[str, tuple[float, float]]:
    """Read Cue's trained parameter ranges (#2569 follow-up to #2460).

    Cue (``cue.py``) is a Speculator neural network, not a grid -- there is
    no ``jnp.clip`` and no #1586 zero-gradient-outside mechanism (see
    :data:`EXTRAPOLATING_SUPPORT`, which this entry is the sole member of).
    Registered anyway: Li et al. 2025's Table 1 documents an explicit
    trained footprint for every one of these five parameters, and a build
    that samples well outside it is extrapolating with no bound-check at
    all -- worth narrowing a free prior to, and worth a warning, even
    though the failure mode differs from a clipped grid.

    The constants themselves, with the paper citation and the unit
    conversion each needed (log U and log n_H match tengri's declared units
    directly; C/O and N/O are converted from Table 1's linear ratio to the
    declared log10 dex), live in ``cue.py`` next to the backend they
    describe, not here.
    """
    from tengri.components.nebular.cue import (
        CUE_TRAINED_LOG_CO,
        CUE_TRAINED_LOG_NH,
        CUE_TRAINED_LOG_NO,
        CUE_TRAINED_LOG_U,
        CUE_TRAINED_LOG_Z_GAS,
    )

    return {
        "neb_logU": CUE_TRAINED_LOG_U,
        "gas_logn": CUE_TRAINED_LOG_NH,
        "neb_logZ_gas": CUE_TRAINED_LOG_Z_GAS,
        "gas_logco": CUE_TRAINED_LOG_CO,
        "gas_logno": CUE_TRAINED_LOG_NO,
    }


#: ``(selector, component name)`` -> accessor for that component's grid support.
#:
#: ``selector`` is the dotted path used by the build grammar (``"agn.disc"``,
#: ``"dust.emission"``). Only template-backed components appear; a component
#: absent from this table is unconstrained, which is the correct default for
#: every closed-form model.
GRID_SUPPORT: dict[tuple[str, str], GridSupportFn] = {
    ("agn.disc", "slone_netzer"): _slone_netzer_support,
    ("agn.disc", "kd18_agnfitter"): _kd18_agnfitter_support,
    ("agn.disc", "kd18_agnfitter_warmindex"): _kd18_agnfitter_warmindex_support,
    ("neb", "cloudy"): _cloudy_grid_support,
    ("neb", "cb19"): _cb19_grid_support,
    ("neb", "mappings"): _mappings_stellar_grid_support,
    ("neb", "mappings_agn"): _mappings_agn_grid_support,
    ("neb", "cue"): _cue_grid_support,
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
# Nebular components registered (#2460): ("neb", "cloudy"), ("neb", "cb19"),
# ("neb", "mappings"), ("neb", "mappings_agn") -- all four clip neb_logU onto
# a grid axis (confirmed by reading each interpolator: CloudyGridBackend's
# triweight/linear lookup, CB19Backend's _frac_idx, MAPPINGS'
# _interp_index_weight), so all four share the #1586 zero-gradient-outside
# mechanism this registry exists to narrow. Only "cloudy" also registers
# neb_logZ_gas: CB19's metallicity axis is log_OH_total and MAPPINGS' is
# ζ_O, neither log10(Z/Zsun), so narrowing them from the shared declaration
# would need a unit conversion this fix does not establish.
#
# Cue ("cue", #2569) IS registered, but its mechanism differs from the four
# above: it is a Speculator neural-network emulator (cue.py), not a grid, so
# nothing clips and a draw beyond the trained range does NOT zero the
# gradient the way grid interpolation does -- the failure mode is silently
# untrustworthy extrapolation, not dead inference signal. It is registered
# anyway against Li et al. 2025's Table 1 trained ranges (cue.py's
# CUE_TRAINED_* constants) for neb_logU, gas_logn, neb_logZ_gas, gas_logco
# and gas_logno, because a declared prior wider than the trained footprint
# is exactly the kind of overhang this registry exists to narrow and flag --
# see EXTRAPOLATING_SUPPORT, which ("neb", "cue") is the sole member of, and
# which describe_clipping/_warn_on_grid_overhang consult to word the warning
# for extrapolation rather than clipping.
#
# The grid support accessor returns empty dict {} if the grid file
# is not found, so model construction does not fail when grids are unavailable.
# dh02_ce01 is deliberately absent: its only grid axis is L_TIR, derived from
# L_absorbed by energy balance rather than set by the user, so no prior can
# overhang it. See _DUST_EMISSION_GRID_AXES for the same reasoning on bosa.


def grid_support(selector: str, name: str) -> dict[str, tuple[float, float]]:
    """Return the grid support of one component, or ``{}`` if it has none.

    Parameters
    ----------
    selector : str
        Dotted selector path, e.g. ``'agn.disc'`` or ``'dust.emission'``.
    name : str
        Component name, e.g. ``'slone_netzer'`` or ``'themis'``.

    Returns
    -------
    support : dict[str, tuple[float, float]]
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
#: A prior written to match a grid axis is normally hand-entered to a handful of
#: decimals, so it can overhang the true bound by a few ulp. Comparing exactly
#: reports that rounding as a defect and then prints a self-contradictory
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
    active : tuple[float, float]
        ``(lo, hi)`` the parameter can actually take -- a prior's bounds, or
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


#: ``(selector, name)`` pairs whose registered support is a smooth model's
#: trained footprint, not a template grid's clipped interpolation axis.
#: ``jnp.clip`` never runs for these -- exploring past the footprint does
#: NOT zero the gradient or freeze the SED (#1586's mechanism is specific to
#: grid interpolation); the model keeps predicting, smoothly but with no
#: guarantee it is still accurate. :func:`describe_clipping` and
#: :func:`check_grid_support` consult this set so the warning text says that
#: instead of the (false, for these) clip claim. The only member today is
#: ``("neb", "cue")``: a Speculator neural network (``cue.py``) registered
#: against its trained parameter ranges (Li et al. 2025, Table 1) for the
#: same reason a grid is registered against its axes -- a declared prior
#: wider than what the model was ever validated on is worth narrowing and
#: flagging -- but the failure mode past the edge is untrustworthy
#: extrapolation, not inertness.
EXTRAPOLATING_SUPPORT: frozenset[tuple[str, str]] = frozenset({("neb", "cue")})


def describe_clipping(
    active: tuple[float, float], grid: tuple[float, float], *, extrapolates: bool = False
) -> str | None:
    """Describe how an active support overhangs a grid, or ``None`` if it fits.

    Returns only the component-agnostic clause, so each caller keeps its own
    framing and existing message wording stays byte-identical.

    Parameters
    ----------
    active : tuple[float, float]
        ``(lo, hi)`` the parameter can actually take.
    grid : tuple[float, float]
        ``(lo, hi)`` covered by the template grid.
    extrapolates : bool
        ``True`` for a ``(selector, name)`` in :data:`EXTRAPOLATING_SUPPORT`:
        the component is a smooth emulator, not a clipped grid, so the
        clause describes untrustworthy extrapolation instead of a frozen,
        bit-identical SED. Default ``False`` (every grid-backed component)
        keeps the original wording byte-identical.

    Returns
    -------
    detail : str or None
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
    footprint = "trained footprint" if extrapolates else "grid extent"
    if a_lo == a_hi:
        fate = (
            "so it is extrapolated"
            if extrapolates
            else "so it is clipped onto the nearest edge node"
        )
        return f"the fixed value {a_lo:g} lies outside the {footprint} {extent}, {fate}"
    if not (math.isfinite(a_lo) and math.isfinite(a_hi)):
        # An unbounded prior (e.g. an untruncated Gaussian) is NOT inert --
        # most of its mass may sit on the grid. Only the tails clip, so say
        # that and do not quote a percentage: the fraction of an infinite
        # support is not informative.
        fate = "are extrapolated past it" if extrapolates else "are clipped onto an edge node"
        return (
            f"its support [{a_lo:g}, {a_hi:g}] is unbounded, so the tails "
            f"beyond the {footprint} {extent} {fate}"
        )
    live = live_fraction(active, grid)
    if live == 0.0:
        fate = (
            "the parameter reaches only untrustworthy extrapolation -- every "
            "value is outside where the model was validated"
            if extrapolates
            else "the parameter is entirely inert -- every value gives the same SED"
        )
        return (
            f"its whole range [{a_lo:g}, {a_hi:g}] lies outside the {footprint} "
            f"{extent}, so {fate}"
        )
    fate = "extrapolated past it" if extrapolates else "silently clipped onto an edge node"
    return (
        f"{100.0 * (1.0 - live):.0f}% of its range [{a_lo:g}, {a_hi:g}] lies "
        f"outside the {footprint} {extent} and is {fate}"
    )


def _cue_logno_shift(param_support: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
    """Range of the default N/O--O/H relation over the reachable ``neb_logZ_gas``.

    ``gas_logno`` is an offset from that relation (#2693), while Cue's trained
    [N/O] range is absolute, so the effective [N/O] a draw reaches is
    ``gas_logno + relation(neb_logZ_gas)``. The relation increases with
    metallicity, so its extremes sit at the metallicity bounds.
    """
    from tengri.components.nebular._default_nitrogen import default_nitrogen_offset

    z_lo, z_hi = param_support.get("neb_logZ_gas", (0.0, 0.0))
    return (float(default_nitrogen_offset(z_lo)), float(default_nitrogen_offset(z_hi)))


#: ``(selector, name, param)`` -> function mapping the reachable parameter
#: ranges to the ``(lo, hi)`` that is ADDED to this parameter's own range to
#: obtain the quantity the registered support bounds. Used where the declared
#: parameter is an offset but the support is absolute.
SUPPORT_SHIFT: dict[tuple[str, str, str], Callable[..., tuple[float, float]]] = {
    ("neb", "cue", "gas_logno"): _cue_logno_shift,
}


def support_shift(
    selector: str, name: str, pname: str, param_support: Mapping[str, tuple[float, float]]
) -> tuple[float, float]:
    """The ``(lo, hi)`` added to ``pname``'s range to get the bounded quantity (0 if none)."""
    fn = SUPPORT_SHIFT.get((selector, name, pname))
    return (0.0, 0.0) if fn is None else fn(param_support)


def check_grid_support(
    selected: Iterable[tuple[str, str]],
    param_support: Mapping[str, tuple[float, float]],
) -> list[tuple[str, str, str, str, tuple[float, float]]]:
    """Find every selected component whose grid cannot cover an active support.

    Parameters
    ----------
    selected : iterable of (str, str)
        ``(selector, name)`` pairs for the components in play, e.g.
        ``[("dust.emission", "themis")]``.
    param_support : mapping of str to (float, float)
        ``{param_name: (lo, hi)}`` the range each parameter can actually take
        -- a prior's bounds, or ``(v, v)`` for a fixed value.

    Returns
    -------
    findings : list of tuple
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
        extrapolates = (selector, name) in EXTRAPOLATING_SUPPORT
        for pname, extent in grid_support(selector, name).items():
            active = param_support.get(pname)
            if active is None:
                continue
            shift = support_shift(selector, name, pname, param_support)
            if shift != (0.0, 0.0):
                eff = (active[0] + shift[0], active[1] + shift[1])
                detail = describe_clipping(eff, extent, extrapolates=extrapolates)
                if detail is not None:
                    detail = (
                        f"the effective range (offset [{active[0]:g}, {active[1]:g}] plus the "
                        f"default relation [{shift[0]:g}, {shift[1]:g}]): {detail}"
                    )
            else:
                detail = describe_clipping(active, extent, extrapolates=extrapolates)
            if detail is not None:
                findings.append((selector, name, pname, detail, extent))
    return findings


__all__ = [
    "EXTRAPOLATING_SUPPORT",
    "GRID_SUPPORT",
    "SUPPORT_SHIFT",
    "GridSupportFn",
    "check_grid_support",
    "describe_clipping",
    "grid_support",
    "is_contained",
    "live_fraction",
    "support_shift",
]
