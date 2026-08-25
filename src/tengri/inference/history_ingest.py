# SPDX-License-Identifier: BSD-3-Clause
"""Simulation history ingestion: SFH / Z(t) tables -> validated arrays (#1677).

The sibling of :mod:`tengri.inference.catalog_ingest`. That module turns an
observed flux table into contiguous validated arrays; this one does the same for
the *records* a simulation supplies (SFR(t) and, optionally, Z(t)) so
:meth:`tengri.Catalog.from_histories` validates once, at the boundary, before
anything compiles.

Four things go wrong silently when a hydro sim or SAM history is handed straight
to the forward model, and each is measured in the #1677 trace:

* **Out-of-grid metallicity clamps.** Metallicity lookups ``jnp.clip`` onto
  ``ssp_lgmet`` (``dsps_wrapper.py``), so ``logzsol = -6`` returns byte-identical
  photometry to the grid edge at ``-2.152``. A simulation reaches primordial
  values at early times as a matter of course, so this is the default case, not
  the edge case.
* **Units.** A snapshot stores metallicity as a mass fraction :math:`Z`, not as
  :math:`\\log_{10}(Z/Z_\\odot)`. ``Z = 2e-4`` is a legal ``logzsol`` value
  (1.0 :math:`Z_\\odot`) and the true value is 0.014 :math:`Z_\\odot`, a factor
  of ~70, accepted without complaint. ``met_unit=`` lets a caller declare the
  convention, but it does not close the hole by itself: a mass fraction is
  *in-grid* when read as log10(Z/Zsun), so the range check cannot see it. The
  discriminator is dynamic range, and it is a separate check.
* **Non-finite nodes.** ``np.any(sfr < 0.0)`` is ``False`` for ``NaN``, so the
  negative-SFR guard admits ``NaN`` unexamined; a ``NaN`` in ``met`` was measured
  to leave the predicted flux bit-identical, i.e. dropped entirely.
* **A tabulated model with no table.** A model built ``met_mode='table'`` and
  given no ``met=`` was accepted at construction and failed inside the forward
  pass on the first ``predict()``, which is the fail-late that ``from_histories``
  exists to prevent.

Nothing here is JAX: ingestion is numpy, eager, and raises Python exceptions.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from tengri.config.exceptions import (
    MetallicityUnitWarning,
    OutOfSSPGridWarning,
    warn_measured,
)
from tengri.parameters.translate import LOG10_ZSUN

__all__ = [
    "MET_UNITS",
    "ON_OUT_OF_GRID",
    "HistoryArrays",
    "ingest_histories",
    "met_to_logzsol",
]

#: Accepted values of ``met_unit``, mapped to a one-line description. The keys
#: are the contract; :func:`met_to_logzsol` is the only place they are consumed.
MET_UNITS = {
    "logzsol": "log10(Z/Zsun), tengri's user-facing metallicity unit",
    "log_z_abs": "log10(Z) absolute, the SSP grid's own convention",
    "z_mass_fraction": "Z, the metal mass fraction, what a simulation snapshot stores",
}

#: Accepted values of ``on_out_of_grid``.
ON_OUT_OF_GRID = ("raise", "warn", "ignore")

#: Metal mass fraction below which ``met_unit='z_mass_fraction'`` refuses to take
#: a logarithm. Primordial gas is genuinely ``Z = 0`` in most simulations, and
#: ``log10(0)`` is ``-inf``, which would reach the grid check as a nonsense
#: number instead of as the modeling decision it actually is.
_Z_FLOOR = 0.0

#: Upper edge of the log10(Z/Zsun) band that reads as a mass fraction. 0.1 dex
#: above solar is 1.26 Zsun, and covers metal mass fractions up to Z = 0.1,
#: above anything a simulation snapshot holds. See
#: :func:`_check_unit_plausibility` for why the band, not the magnitude, is the
#: discriminator.
_MASS_FRACTION_BAND = 0.1


class HistoryArrays(NamedTuple):
    """Contiguous validated simulation histories, on one common convention.

    Attributes
    ----------
    t_gyr : ndarray, shape (N, n_t)
        Cosmic time [Gyr], strictly increasing along axis 1. A shared 1-D grid
        has already been broadcast to per-galaxy rows.
    sfr : ndarray, shape (N, n_t)
        Star formation rate [Msun/yr] at those times, finite and non-negative.
    met : ndarray, shape (N, n_t) or None
        **Stellar** metallicity history as log10(Z/Zsun), the Z each generation
        of stars formed from, which selects the SSP templates. Converted at
        ingest so exactly one convention leaves this module. ``None`` when no
        history was supplied.
    met_gas : ndarray, shape (N,) or None
        **Gas-phase** metallicity at the observed epoch, log10(Z/Zsun), the Z
        of the ionized gas, which drives nebular emission. A per-galaxy scalar
        rather than a history: nebular emission is powered by stars younger than
        ~10 Myr, so only the present-day value is observable. ``None`` when not
        supplied, in which case the model's own ``neb_logZ_gas`` applies.
    n_galaxies : int
        Number of galaxies, N.
    n_t : int
        Number of history nodes.
    mass_formed : ndarray, shape (N,)
        Stellar mass formed by trapezoidal integration of the history [Msun].
        Carried because the out-of-grid diagnostic is mass-weighted, and a
        caller comparing against a simulation's own catalog mass wants the
        number this ingest actually saw.
    """

    t_gyr: np.ndarray
    sfr: np.ndarray
    met: np.ndarray | None
    n_galaxies: int
    n_t: int
    mass_formed: np.ndarray
    met_gas: np.ndarray | None = None


def met_to_logzsol(met, met_unit):
    """Convert a metallicity history to log10(Z/Zsun).

    Parameters
    ----------
    met : array_like
        Metallicity values in ``met_unit``. Any shape.
    met_unit : str
        One of :data:`MET_UNITS`.

    Returns
    -------
    ndarray
        The same shape, as log10(Z/Zsun) [dimensionless].

    Raises
    ------
    ValueError
        If ``met_unit`` is unknown, or if ``met_unit='z_mass_fraction'`` and any
        value is non-positive.

    Notes
    -----
    **JIT-compatible**: no, numpy, and it raises on bad input by design.

    The absolute offset is ``LOG10_ZSUN = -1.848`` (Asplund 2009, Zsun = 0.0142),
    the same constant ``_tabulated_lgmet_on_ssp_ages`` adds when it lifts
    ``met_history`` onto the SSP grid, so a round trip through this function and
    back is exact:

    .. math::

        \\log_{10}(Z/Z_\\odot) = \\log_{10} Z - \\log_{10} Z_\\odot

    where :math:`Z` is the metal mass fraction [dimensionless] and
    :math:`\\log_{10} Z_\\odot = -1.848`.

    Per-library solar references differ (BC03/Padova 0.0190, PARSEC 0.0152,
    BASTI 0.0200; see ``LOG10_ZSUN_BY_LIBRARY``). This function deliberately
    uses the one constant the forward model uses, so ingest and forward agree;
    a caller pinning a cross-code comparison should pass ``log_z_abs`` and do
    its own offset.
    """
    if met_unit not in MET_UNITS:
        raise ValueError(
            f"met_unit={met_unit!r} is not a metallicity unit. Valid: "
            f"{', '.join(repr(u) for u in MET_UNITS)}. "
            + " ".join(f"{u!r} is {d}." for u, d in MET_UNITS.items())
        )

    met = np.asarray(met, dtype=float)
    if met_unit == "logzsol":
        return met
    if met_unit == "log_z_abs":
        return met - LOG10_ZSUN

    # z_mass_fraction: take the log, but refuse the primordial zero rather than
    # let -inf reach the grid check disguised as a metallicity.
    non_positive = met <= _Z_FLOOR
    if np.any(non_positive):
        n_bad = int(non_positive.sum())
        worst = float(np.min(met))
        raise ValueError(
            f"met_unit='z_mass_fraction' but {n_bad} of {met.size} values are "
            f"<= 0 (minimum {worst:g}); log10 is undefined there. Primordial gas "
            f"genuinely has Z = 0 in most simulations, so this is a modeling "
            f"decision, not a data error: choose a metallicity floor and apply "
            f"it yourself, e.g. met=np.maximum(Z, 1e-6). Picking one for you "
            f"would put stars at a metallicity you never chose."
        )
    return np.log10(met) - LOG10_ZSUN


def _check_unit_plausibility(logzsol, met_unit):
    """Flag a history that reads like a mass fraction taken for log10(Z/Zsun).

    Parameters
    ----------
    logzsol : ndarray, shape (N, n_t)
        The **converted** history [log10(Z/Zsun)].
    met_unit : str
        The unit the caller declared, quoted back in the message.

    Warns
    -----
    MetallicityUnitWarning
        When every node lies in ``(0, _MASS_FRACTION_BAND)``.

    Notes
    -----
    **JIT-compatible**: no, numpy diagnostic at ingest.

    This exists because ``met_unit=`` alone does not close the units hole, which
    was measured: the SSP-grid check cannot catch a mass fraction, since a small
    positive number is a legal near-solar log10(Z/Zsun) and sits inside every
    grid. Dynamic range is the discriminator, not magnitude.

    Running on the converted values is what keeps this free of false alarms for
    a caller who did declare ``z_mass_fraction``, those convert to roughly
    :math:`-2 \\ldots 0` and are nowhere near the band.
    """
    finite = logzsol[np.isfinite(logzsol)]
    if finite.size == 0 or not np.all((finite > 0.0) & (finite < _MASS_FRACTION_BAND)):
        return
    lo, hi = float(finite.min()), float(finite.max())
    warn_measured(
        f"every metallicity node lies in ({lo:g}, {hi:g}) log10(Z/Zsun), a span of "
        f"{hi - lo:.3f} dex entirely just above solar, read as met_unit={met_unit!r}. "
        f"That is the signature of a metal mass fraction Z taken for log10(Z/Zsun): "
        f"chemical enrichment moves Z(t) by orders of magnitude over cosmic time, so a "
        f"real history is not this flat. As log10(Z/Zsun) these values mean "
        f"{10.0**lo:.2f}-{10.0**hi:.2f} Zsun, which is in-grid and therefore invisible "
        f"to the out-of-grid check (#1677). If they are mass fractions, pass "
        f"met_unit='z_mass_fraction'; if the history really is this flat, filter "
        f"MetallicityUnitWarning.",
        MetallicityUnitWarning,
        logzsol_min=lo,
        logzsol_max=hi,
        span_dex=hi - lo,
        stacklevel=4,
    )


def _mass_weights(t_gyr, sfr):
    """Trapezoidal stellar-mass weight carried by each history node [Msun].

    Parameters
    ----------
    t_gyr : ndarray, shape (N, n_t)
        Cosmic time [Gyr], strictly increasing along axis 1.
    sfr : ndarray, shape (N, n_t)
        Star formation rate [Msun/yr].

    Returns
    -------
    ndarray, shape (N, n_t)
        Mass [Msun] attributed to each node, summing to the trapezoidal integral
        :math:`\\int \\mathrm{SFR}\\,\\mathrm{d}t` over the tabulated span.

    Notes
    -----
    **JIT-compatible**: no, numpy, diagnostic only, never on the forward path.

    .. math::

        w_i = \\mathrm{SFR}_i \\cdot \\tfrac{1}{2}(t_{i+1} - t_{i-1}) \\cdot 10^{9}

    with one-sided differences at the two ends. :math:`t` is cosmic time [Gyr],
    :math:`\\mathrm{SFR}` is [Msun/yr], and the :math:`10^{9}` converts Gyr to yr
    so :math:`w` is [Msun].

    This is the trapezoid rule regrouped per node rather than per interval, which
    is what lets the out-of-grid report say *how much stellar mass* sits on the
    offending nodes instead of merely how many nodes there are. A single node is
    a degenerate history with no span, so its weight is zero.
    """
    if t_gyr.shape[1] < 2:
        return np.zeros_like(sfr)
    spans = np.empty_like(t_gyr)
    spans[:, 1:-1] = 0.5 * (t_gyr[:, 2:] - t_gyr[:, :-2])
    spans[:, 0] = 0.5 * (t_gyr[:, 1] - t_gyr[:, 0])
    spans[:, -1] = 0.5 * (t_gyr[:, -1] - t_gyr[:, -2])
    return sfr * spans * 1e9


def _require_finite(name, arr, unit):
    """Raise naming the first offending galaxy and node, not just the array."""
    bad = ~np.isfinite(arr)
    if not np.any(bad):
        return
    idx = np.argwhere(bad)
    first = tuple(int(i) for i in idx[0])
    where = f"galaxy {first[0]}, node {first[1]}" if len(first) == 2 else f"index {first[0]}"
    raise ValueError(
        f"{name} has {int(bad.sum())} non-finite entries {unit}; first at {where}. "
        f"NaN survives every comparison silently, a NaN SFR passes the "
        f"non-negativity check, and a NaN metallicity node was measured to leave "
        f"the predicted flux bit-identical, i.e. dropped without a word. Replace "
        f"or drop those nodes before ingest."
    )


def _check_ssp_grid(logzsol, mass_w, ssp_lgmet, policy):
    """Compare a converted Z history against the SSP grid it will be read on.

    Parameters
    ----------
    logzsol : ndarray, shape (N, n_t)
        Metallicity history [log10(Z/Zsun)].
    mass_w : ndarray, shape (N, n_t)
        Per-node stellar mass [Msun] from :func:`_mass_weights`, used only to
        report how much of the history the clamp would touch.
    ssp_lgmet : array_like, shape (n_met,)
        The SSP library's metallicity grid, absolute log10(Z).
    policy : {"raise", "warn", "ignore"}
        What to do when nodes fall outside.

    Raises
    ------
    ValueError
        If ``policy='raise'`` and any node is outside the grid.

    Warns
    -----
    OutOfSSPGridWarning
        If ``policy='warn'`` and any node is outside the grid.

    Notes
    -----
    **JIT-compatible**: no, numpy validation, eager by design.

    Every node is checked, including nodes where the SFR is zero. That looks
    over-strict and is not: ``_tabulated_lgmet_on_ssp_ages`` **interpolates**
    Z(t) onto the SSP age grid, so a wild value at a node that formed no stars
    still drags the interpolated metallicity at neighboring ages that did. The
    mass-weighted share is therefore reported, never used as a filter.

    This raises where the scalar-parameter check
    (``SEDModel._validate_metallicity_bounds``, #442) only warns. The two are not
    inconsistent: that one guards a *prior* at build time, where a chain touching
    the edge is a sampling artifact and a fixture SSP may legitimately ship a
    grid in log10(Z/Zsun) already, whereas a history is a *record* arriving at
    ingest, where refusing is the whole contract of ``from_histories``. The
    ``on_out_of_grid='warn'`` escape hatch restores the #442 behavior for anyone
    on such a grid.
    """
    if policy == "ignore" or ssp_lgmet is None:
        return
    lgmet = np.asarray(ssp_lgmet, dtype=float)
    if lgmet.size == 0:
        return

    grid_lo = float(lgmet.min()) - LOG10_ZSUN
    grid_hi = float(lgmet.max()) - LOG10_ZSUN
    outside = (logzsol < grid_lo) | (logzsol > grid_hi)
    if not np.any(outside):
        return

    n_out = int(outside.sum())
    n_total = int(logzsol.size)
    total_mass = float(mass_w.sum())
    mass_frac = float(mass_w[outside].sum() / total_mass) if total_mass > 0.0 else 0.0

    # The worst offender by distance from the grid, named so the reader can go
    # look at that galaxy rather than re-deriving which one it was.
    excess = np.where(outside, np.maximum(grid_lo - logzsol, logzsol - grid_hi), -np.inf)
    g, k = np.unravel_index(int(np.argmax(excess)), logzsol.shape)
    worst = float(logzsol[g, k])

    detail = (
        f"{n_out} of {n_total} Z(t) nodes fall outside the SSP metallicity grid "
        f"[{grid_lo:.3f}, {grid_hi:.3f}] log10(Z/Zsun) (absolute grid log10(Z) in "
        f"[{lgmet.min():.3f}, {lgmet.max():.3f}]). Those nodes carry {mass_frac:.1%} "
        f"of the stellar mass formed; the worst is galaxy {g}, node {k}, at "
        f"logzsol={worst:.3f}. The metallicity lookup clips to the grid edge, so "
        f"these would produce a smooth, plausible and wrong SED with no other "
        f"sign (#442, #1677)."
    )

    if policy == "raise":
        raise ValueError(
            detail + " Clip the history yourself (np.clip), load an SSP whose "
            "grid covers it, or pass on_out_of_grid='warn' to accept the clamp. "
            "Check met_unit= first: a metal mass fraction read as log10(Z/Zsun) "
            "lands out of grid exactly like this."
        )
    warn_measured(
        detail + " Accepted because on_out_of_grid='warn'.",
        OutOfSSPGridWarning,
        n_outside=n_out,
        n_nodes=n_total,
        mass_fraction_outside=mass_frac,
        worst_logzsol=worst,
        grid_lo_zsol=grid_lo,
        grid_hi_zsol=grid_hi,
        stacklevel=4,
    )


def ingest_histories(
    *,
    t_gyr,
    sfr,
    met=None,
    met_gas=None,
    met_unit="logzsol",
    on_out_of_grid="raise",
    ssp_lgmet=None,
) -> HistoryArrays:
    """Validate simulation histories and put them on one convention (#1677).

    Parameters
    ----------
    t_gyr : array_like, shape (n_t,) or (N, n_t)
        Cosmic time [Gyr], strictly increasing. A 1-D grid is shared by every
        galaxy and broadcast.
    sfr : array_like, shape (N, n_t)
        Star formation rate [Msun/yr]. Finite and non-negative.
    met : array_like, shape (n_t,) or (N, n_t), optional
        **Stellar** metallicity history in ``met_unit`` at the same nodes, the
        Z each generation of stars formed from. A 1-D history is shared and
        broadcast, the same way ``t_gyr`` is: one chemical-evolution track
        across many mass scalings is a common simulation case.
    met_gas : array_like, shape (N,), (n_t,) or (N, n_t), optional
        **Gas-phase** metallicity in ``met_unit``, the Z of the ionized gas that
        drives nebular emission. A separate physical quantity from ``met``, and
        settable independently. Given as a track, the last node is taken as the
        observed epoch, nebular emission comes from stars younger than ~10 Myr,
        so only the present-day value is observable.
    met_unit : str, default "logzsol"
        The unit ``met`` **and** ``met_gas`` arrive in. One of :data:`MET_UNITS`.
        A snapshot stores both the same way, so one declaration covers both.
    on_out_of_grid : {"raise", "warn", "ignore"}, default "raise"
        What to do when a metallicity node falls outside the SSP grid, where the
        lookup would silently clip.
    ssp_lgmet : array_like, shape (n_met,), optional
        The SSP library's absolute log10(Z) grid. When ``None`` the grid check is
        skipped; there is nothing to check against.

    Returns
    -------
    HistoryArrays
        Broadcast, validated, and converted to log10(Z/Zsun).

    Raises
    ------
    ValueError
        If a shape disagrees, a value is non-finite, ``t_gyr`` is not strictly
        increasing, an SFR is negative, ``met_unit`` or ``on_out_of_grid`` is
        unknown, or a metallicity node is off-grid under ``on_out_of_grid='raise'``.

    Warns
    -----
    OutOfSSPGridWarning
        Under ``on_out_of_grid='warn'``, once, carrying the measured values.
    MetallicityUnitWarning
        When the converted history reads like a metal mass fraction mistaken for
        log10(Z/Zsun), the case ``on_out_of_grid`` is structurally unable to
        see, because such values are in-grid.

    Notes
    -----
    **JIT-compatible**: no, eager numpy validation, run once at construction.

    Examples
    --------
    >>> h = ingest_histories(t_gyr=t, sfr=sfr, met=Z, met_unit="z_mass_fraction")
    >>> h.met.shape == h.sfr.shape
    True
    """
    if on_out_of_grid not in ON_OUT_OF_GRID:
        raise ValueError(
            f"on_out_of_grid={on_out_of_grid!r} is not a policy. Valid: "
            f"{', '.join(repr(p) for p in ON_OUT_OF_GRID)}."
        )

    sfr = np.asarray(sfr, dtype=float)
    if sfr.ndim != 2:
        raise ValueError(f"sfr must be (N, n_t); got shape {sfr.shape}.")
    n_galaxies, n_t = sfr.shape

    t_gyr = _broadcast_history("t_gyr", t_gyr, n_galaxies, n_t)

    _require_finite("t_gyr", t_gyr, "[Gyr]")
    _require_finite("sfr", sfr, "[Msun/yr]")

    if not np.all(np.diff(t_gyr, axis=1) > 0.0):
        raise ValueError(
            "t_gyr must be strictly increasing along the time axis (cosmic "
            "time [Gyr], not lookback). A non-monotonic grid interpolates to "
            "garbage without raising downstream."
        )
    if np.any(sfr < 0.0):
        raise ValueError(
            "sfr has negative entries [Msun/yr]. A negative SFR subtracts "
            "stellar mass in the age-weight integral, silently."
        )

    mass_w = _mass_weights(t_gyr, sfr)

    logzsol = None
    if met is not None:
        # Finiteness is checked on the values as supplied, so the message quotes
        # the caller's own numbers rather than a converted stand-in for them.
        met_raw = _broadcast_history("met", met, n_galaxies, n_t)
        _require_finite("met", met_raw, f"[{met_unit}]")
        logzsol = met_to_logzsol(met_raw, met_unit)
        _check_unit_plausibility(logzsol, met_unit)
        _check_ssp_grid(logzsol, mass_w, ssp_lgmet, on_out_of_grid)

    gas = None
    if met_gas is not None:
        # Not checked against ``ssp_lgmet``: the gas-phase value is read on the
        # *nebular* backend's own grid (Cue, CLOUDY), a different axis with a
        # different span, so the stellar grid would be the wrong yardstick.
        gas_raw = _present_day("met_gas", met_gas, n_galaxies, n_t)
        _require_finite("met_gas", gas_raw, f"[{met_unit}]")
        gas = met_to_logzsol(gas_raw, met_unit)

    return HistoryArrays(
        t_gyr=t_gyr,
        sfr=sfr,
        met=logzsol,
        n_galaxies=n_galaxies,
        n_t=n_t,
        mass_formed=mass_w.sum(axis=1),
        met_gas=gas,
    )


def _present_day(name, value, n_galaxies, n_t):
    """A per-galaxy scalar, taken from a track's last node if a track is given.

    Parameters
    ----------
    name : str
        Argument name, for the error messages.
    value : array_like, shape (N,), (n_t,) or (N, n_t)
        Either the per-galaxy value already, or a full track to read the
        observed epoch off.
    n_galaxies, n_t : int
        The catalog's shape, from ``sfr``.

    Returns
    -------
    ndarray, shape (N,)

    Raises
    ------
    ValueError
        If the shape is not one of the three, or if ``N == n_t`` makes a 1-D
        input ambiguous.

    Notes
    -----
    **JIT-compatible**: no, numpy, at ingest.

    Accepting a track is the convenience that matters for simulation data, where
    gas-phase metallicity is stored as a time series exactly like the SFH. The
    **last** node is the observed epoch because ``t_gyr`` is cosmic time and
    ascending, the same orientation ``from_histories`` requires of the SFH.

    A 1-D input is ambiguous when ``N == n_t``: it could be one value per galaxy
    or one shared track. That is refused rather than guessed, the two readings
    give different physics, and a silent choice between them is unrecoverable.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        if arr.shape != (n_galaxies, n_t):
            raise ValueError(
                f"{name} as a track must be (N, n_t) = ({n_galaxies}, {n_t}); got {arr.shape}."
            )
        return arr[:, -1].copy()
    if arr.ndim == 0:
        return np.full(n_galaxies, float(arr))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be (N,), (n_t,) or (N, n_t); got shape {arr.shape}.")
    if n_galaxies == n_t and arr.shape[0] == n_galaxies:
        raise ValueError(
            f"{name} is 1-D with {n_galaxies} entries and the catalog has "
            f"N == n_t == {n_galaxies}, so it could equally be one value per "
            f"galaxy or one shared track, the two mean different things and "
            f"neither can be recovered from the other afterwards. Pass it as "
            f"(N, n_t) to mean a track, or reshape to (N, 1) to mean per-galaxy."
        )
    if arr.shape[0] == n_galaxies:
        return arr.copy()
    if arr.shape[0] == n_t:
        return np.full(n_galaxies, float(arr[-1]))
    raise ValueError(
        f"{name} has {arr.shape[0]} entries, which is neither N={n_galaxies} "
        f"(one per galaxy) nor n_t={n_t} (a shared track)."
    )


def _broadcast_history(name, value, n_galaxies, n_t):
    """A shared 1-D ``(n_t,)`` history, or a per-galaxy ``(N, n_t)`` one."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n_t:
            raise ValueError(
                f"{name} and sfr disagree on n_t: {name} has {arr.shape[0]}, "
                f"sfr has {n_t}. They index the same history nodes."
            )
        return np.broadcast_to(arr, (n_galaxies, n_t)).copy()
    if arr.ndim != 2:
        raise ValueError(f"{name} must be (n_t,) or (N, n_t); got shape {arr.shape}.")
    if arr.shape != (n_galaxies, n_t):
        raise ValueError(
            f"{name} must match sfr's shape (N, n_t) = ({n_galaxies}, {n_t}); got {arr.shape}."
        )
    return arr
