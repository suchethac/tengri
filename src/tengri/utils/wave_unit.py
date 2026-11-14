# SPDX-License-Identifier: BSD-3-Clause
"""One wavelength-unit spelling at every raw-array boundary, and plausibility refusals.

Wavelength arrays enter tengri through several public entries (custom filter
curves, DSPS transmission curves, :class:`~tengri.observation.spectroscopy.Spectroscopy`).
Every such entry takes ``wave_unit=`` with one spelling set, converts the array
to Angstrom once at ingestion, and then refuses a range that is unambiguous
nonsense for Angstrom. Timescale parameters (``*_myr``, ``*_gyr``, ``*_yr``)
are bounded the same way when a :class:`~tengri.parameters.parameters.Parameters`
is built.

Wavelength unit
---------------
Accepted names, case-insensitive: ``"AA"`` / ``"angstrom"`` (factor 1),
``"nm"`` (factor 10), ``"um"`` / ``"micron"`` (factor 1e4). Frequency names
``"Hz"``, ``"kHz"``, ``"MHz"``, ``"GHz"`` convert by lambda = c / nu, and energy
names ``"eV"``, ``"keV"`` by lambda = hc / E. Those two families reverse the
order of the array, so :func:`ascending_in_angstrom` reorders any aligned
transmission or flux array with it.

Measured plausibility range
---------------------------
Measured over all 475 filters that :func:`tengri.observation.filters.list_available_filters`
lists and that load (``load_filter``), on 2026-10-14:

* shortest: 0.1569 Å (``nustar_full``), so the floor is ``0.01`` Å
  (0.1569 / 10 = 0.0157, rounded down to one significant figure).
* longest: 1.3091e8 Å (``planck_lfi_030``, about 1.3 cm), inside the ceiling.

The ceiling is the forward model's radio extent, :data:`~tengri.utils.wavelength.RADIO_WAVE_MAX`
(3e11 Å), so a 1.4 GHz band registers. A curve or spectrum whose range lies wholly
outside ``[WAVE_FLOOR_AA, WAVE_CEILING_AA]`` (``0.01`` Å to ``3e11`` Å) is refused
with :class:`ValueError`. A range wholly inside the extreme-UV gap, ``100`` Å to
``1340`` Å, is not refused: it is physically empty of photometric bandpasses, so it
warns that the values are most likely nanometers (see :func:`check_wavelength_plausible`).
A custom band whose unit was not stated and that lies wholly in ``0.1`` Å to ``100`` Å
warns that it may be keV.

Timescales
----------
Bounds are checked once at build, on the declared distribution bounds of
every parameter whose name ends in ``_myr``, ``_gyr`` or ``_yr``. The
checks run in ``Parameters.__init__``, never per draw and never inside JIT.
The check is a unit diagnosis, so two kinds of value are outside its reach:
an exact zero, which is the same number in every unit and is the declared
"ongoing" value of the cessation lookbacks (``sfh_const_end_gyr`` and its
kin, ``0 = star formation continues``), and a log-scale name such as
``sfh_burst_log_tpeak_myr`` (units ``log10(Myr)``), whose value is not in
the suffix's unit at all. Negative values are refused with their own message.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from tengri.config.exceptions import ParameterError, warn_measured
from tengri.utils.physics_constants import C_AA, HC_KEV_ANGSTROM
from tengri.utils.wavelength import RADIO_WAVE_MAX

__all__ = [
    "EUV_GAP_BLUE_EDGE_AA",
    "EUV_GAP_RED_EDGE_AA",
    "TIMESCALE_BOUNDS",
    "WAVE_CEILING_AA",
    "WAVE_FLOOR_AA",
    "WAVE_UNIT_FACTORS",
    "ascending_in_angstrom",
    "check_timescale",
    "check_timescales",
    "check_wavelength_plausible",
    "to_angstrom",
]

#: Multiply a wavelength in the named unit by this factor to reach Angstrom.
#: Keys are lower-case; lookup is case-insensitive.
WAVE_UNIT_FACTORS: dict[str, float] = {
    "aa": 1.0,
    "angstrom": 1.0,
    "nm": 10.0,
    "um": 1.0e4,
    "micron": 1.0e4,
}

#: Shortest plausible wavelength for an Angstrom array, in Angstrom.
#: Measured floor: 0.1569 Å (``nustar_full``) / 10, rounded down.
WAVE_FLOOR_AA: float = 0.01

#: Longest plausible wavelength for an Angstrom array, in Angstrom: the forward model's
#: radio extent, shared with the grid builder so a band the model can predict is accepted.
#: Measured maximum: 1.3091e8 Å (``planck_lfi_030``), inside this ceiling.
WAVE_CEILING_AA: float = RADIO_WAVE_MAX

#: Red edge of the extreme-UV gap, in Angstrom: the blue edge of GALEX FUV, the
#: bluest bandpass tengri ships. A curve lying wholly in 100-1340 Å is almost
#: always nanometers, so it warns rather than refuses.
EUV_GAP_BLUE_EDGE_AA: float = 100.0
EUV_GAP_RED_EDGE_AA: float = 1340.0

#: A custom band with no stated unit that lies wholly in this range (Angstrom) may be
#: keV: 1 keV is 12.4 Å and 124 keV is 0.1 Å. Warns, never refuses, since 0.1-100 Å is
#: a valid X-ray wavelength range.
KEV_LOOKING_BLUE_EDGE_AA: float = 0.1
KEV_LOOKING_RED_EDGE_AA: float = EUV_GAP_BLUE_EDGE_AA

#: Hertz per frequency unit (``nu [Hz] = value * factor``). Lower-case keys.
_HZ_PER_UNIT: dict[str, float] = {"hz": 1.0, "khz": 1.0e3, "mhz": 1.0e6, "ghz": 1.0e9}

#: keV per energy unit (``E [keV] = value * factor``). Lower-case keys.
_KEV_PER_UNIT: dict[str, float] = {"ev": 1.0e-3, "kev": 1.0}

#: Plausible bounds in the unit named by a timescale suffix: ``(lo, hi)``.
TIMESCALE_BOUNDS: dict[str, tuple[float, float]] = {
    "_myr": (1.0e-3, 2.0e4),
    "_gyr": (1.0e-6, 20.0),
    "_yr": (1.0, 2.0e10),
}


#: Canonical spellings shown to users. Lookup accepts any case of each.
_DISPLAY_UNITS: tuple[str, ...] = (
    "AA",
    "angstrom",
    "nm",
    "um",
    "micron",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "eV",
    "keV",
)


def _accepted_units() -> str:
    """Return the canonical accepted unit names as a comma-separated string."""
    return ", ".join(repr(name) for name in _DISPLAY_UNITS)


def to_angstrom(wave: np.ndarray, wave_unit: str) -> np.ndarray:
    """Convert a wavelength array to Angstrom, validating the unit name.

    Parameters
    ----------
    wave : ndarray, shape (n,)
        Values in *wave_unit*.
    wave_unit : str
        One of ``"AA"``, ``"angstrom"``, ``"nm"``, ``"um"``, ``"micron"``
        (wavelength), ``"Hz"``, ``"kHz"``, ``"MHz"``, ``"GHz"`` (frequency), or
        ``"eV"``, ``"keV"`` (energy), case-insensitive.

    Returns
    -------
    ndarray, shape (n,)
        Wavelengths [Angstrom], in the same order as *wave*. The input is returned
        unchanged when the unit is already Angstrom. Frequency and energy units give
        a descending wavelength array for an ascending input; use
        :func:`ascending_in_angstrom` to reorder the aligned arrays with it.

    Raises
    ------
    ValueError
        If *wave_unit* is not an accepted name. The message lists the accepted
        names.
    """
    key = str(wave_unit).strip().lower()
    if key in WAVE_UNIT_FACTORS:
        factor = WAVE_UNIT_FACTORS[key]
        return wave if factor == 1.0 else wave * factor
    if key in _HZ_PER_UNIT:
        return C_AA / (wave * _HZ_PER_UNIT[key])
    if key in _KEV_PER_UNIT:
        return HC_KEV_ANGSTROM / (wave * _KEV_PER_UNIT[key])
    raise ValueError(
        f"Unknown wave_unit={wave_unit!r}. Accepted names (case-insensitive): {_accepted_units()}."
    )


def ascending_in_angstrom(
    wave: np.ndarray, wave_unit: str, *aligned: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Convert *wave* to Angstrom and reorder *aligned* arrays so wavelength ascends.

    Parameters
    ----------
    wave : ndarray, shape (n,)
        Values in *wave_unit*.
    wave_unit : str
        Any name accepted by :func:`to_angstrom`.
    *aligned : ndarray, shape (n,)
        Arrays index-matched to *wave*, such as a transmission or flux column.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, *aligned)``. When the converted wavelength descends, which
        frequency and energy units always do for ascending input, the wavelength
        and every aligned array are reversed together.
    """
    wave_aa = np.asarray(to_angstrom(np.asarray(wave, dtype=np.float64), wave_unit))
    arrays = tuple(np.asarray(a) for a in aligned)
    if wave_aa.size > 1 and wave_aa[0] > wave_aa[-1]:
        wave_aa = wave_aa[::-1]
        arrays = tuple(a[::-1] for a in arrays)
    return (wave_aa, *arrays)


def check_wavelength_plausible(
    wave: np.ndarray,
    name: str,
    *,
    source: str,
    wave_unit: str | None = "AA",
) -> None:
    """Refuse or warn on a wavelength range that is implausible in Angstrom.

    Parameters
    ----------
    wave : array_like, shape (n,)
        Wavelengths [Angstrom], after conversion by :func:`to_angstrom`.
    name : str
        Filter or spectrum name for the message.
    source : str
        Source description (e.g. ``"custom filter"``, ``"DSPS transmission curve"``).
    wave_unit : str or None, optional
        The unit the caller stated. The extreme-UV ambiguity warning applies
        only when this is Angstrom, because a stated nanometer or micron
        input has already been converted and its range cannot be ambiguous.
        ``None`` means no unit was stated: the range is then also checked for
        keV-looking values. Default: ``"AA"``.

    Raises
    ------
    ValueError
        If the range lies wholly below :data:`WAVE_FLOOR_AA` (``0.01`` Å) or
        wholly above :data:`WAVE_CEILING_AA` (``3e11`` Å). Both are
        unambiguous nonsense for Angstrom.

    Warns
    -----
    UserWarning
        If the range lies wholly inside the extreme-UV gap, 100-1340 Å, and
        the unit was Angstrom or not stated. Such a range is most likely nanometers.
        If the unit was not stated and the range lies wholly in 0.1-100 Å, the
        values may be keV.

    Notes
    -----
    Measured floor and ceiling, with the shipped data behind them, are in the
    module docstring. Micron-scale confusion is not detectable from the range
    alone: a 0.3-1.0 Å curve is a real NuSTAR band, so it is accepted. The
    refusal is a backstop, not a substitute for ``wave_unit=``.
    """
    wave_arr = np.asarray(wave, dtype=np.float64)
    if wave_arr.size < 1:
        return

    wave_min = float(np.min(wave_arr))
    wave_max = float(np.max(wave_arr))

    if wave_max < WAVE_FLOOR_AA or wave_min > WAVE_CEILING_AA:
        raise ValueError(
            f"{source} '{name}' spans {wave_min:.4g}-{wave_max:.4g} Å, outside the "
            f"plausible Angstrom range [{WAVE_FLOOR_AA:g}, {WAVE_CEILING_AA:.0e}] Å "
            f"(measured over the shipped filters: 0.157-1.31e8 Å). Likely unit: "
            f"micron if the values are below {WAVE_FLOOR_AA:g} Å, or a frequency or "
            f"energy column if they are above {WAVE_CEILING_AA:.0e} Å. "
            f"Fix: pass wave_unit='um' for micron input, wave_unit='nm' for nanometer "
            f"input, wave_unit='GHz' or 'keV' for frequency or energy input, "
            f"or check the column."
        )

    unit_stated = wave_unit is not None
    unit_is_angstrom = not unit_stated or wave_unit.strip().lower() in ("aa", "angstrom")
    if unit_is_angstrom and (wave_min >= EUV_GAP_BLUE_EDGE_AA and wave_max <= EUV_GAP_RED_EDGE_AA):
        warn_measured(
            f"{source} '{name}' spans {wave_min:.0f}-{wave_max:.0f} Å, which lies "
            f"entirely in the extreme-UV ({EUV_GAP_BLUE_EDGE_AA:.0f}-"
            f"{EUV_GAP_RED_EDGE_AA:.0f} Å). "
            f"The ISM is opaque there, so these are most likely nanometers: "
            f"{wave_min:.0f}-{wave_max:.0f} nm -> {wave_min * 10:.0f}-{wave_max * 10:.0f} Å. "
            f"Pass wave_unit='nm' to convert at the boundary.",
            UserWarning,
            stacklevel=4,
            wave_min_aa=wave_min,
            wave_max_aa=wave_max,
        )

    if not unit_stated and (
        wave_min >= KEV_LOOKING_BLUE_EDGE_AA and wave_max <= KEV_LOOKING_RED_EDGE_AA
    ):
        warn_measured(
            f"{source} '{name}' spans {wave_min:.4g}-{wave_max:.4g} Å with no wave_unit "
            f"stated. That is a valid X-ray wavelength range, but the same numbers are "
            f"a keV band if they were meant as energies. If these are keV, pass "
            f"wave_unit='keV'; if they are Angstrom, pass wave_unit='AA' to state it.",
            UserWarning,
            stacklevel=4,
            wave_min_aa=wave_min,
            wave_max_aa=wave_max,
        )


def check_timescale(value: float | None, name: str) -> None:
    """Refuse a timescale value outside its plausible range for its suffix.

    Parameters
    ----------
    value : float or None
        Timescale value in the unit named by the suffix of *name*. ``None``
        and non-finite values are not checked here.
    name : str
        Parameter name. Only names ending in ``_myr``, ``_gyr`` or ``_yr``
        are checked; any other name returns immediately.

    Raises
    ------
    ParameterError
        If *value* is negative, or positive and outside the plausible range for
        the suffix (see :data:`TIMESCALE_BOUNDS`). The message names the
        parameter, value and unit. An exact zero always passes: it is the same
        number in every unit, so it cannot be a unit mistake, and it is the
        declared "ongoing" value of the cessation lookbacks. A log-scale name
        (a ``log`` or ``log10`` token before the suffix) is not checked.
    """
    unit = _timescale_unit(name)
    if unit is None or value is None:
        return
    val = float(value)
    if not math.isfinite(val) or val == 0.0:
        return
    label = unit.lstrip("_")
    if val < 0.0:
        raise ParameterError(
            f"Parameter {name!r} = {val:g} {label} is negative; a timescale is "
            f"non-negative (0 means 'ongoing' where the parameter allows it)."
        )
    lo, hi = TIMESCALE_BOUNDS[unit]
    if not lo <= val <= hi:
        raise ParameterError(
            f"Parameter {name!r} = {val:g} {label} is outside the plausible range "
            f"[{lo:g}, {hi:g}] {label}. Check the unit of the value: the suffix "
            f"'{unit}' fixes it."
        )


def check_timescales(distributions: Mapping[str, object]) -> None:
    """Check every declared bound of every timescale parameter.

    Parameters
    ----------
    distributions : mapping of str to Distribution
        ``Parameters._distributions``: name to distribution, each with a
        ``bounds`` attribute ``(lo, hi)``. Fixed distributions contribute their
        single value. Unbounded (``None``, non-finite) ends are skipped.

    Raises
    ------
    ParameterError
        If any bound of a timescale parameter is implausible.
    """
    for name, dist in distributions.items():
        if _timescale_unit(name) is None:
            continue
        for bound in getattr(dist, "bounds", ()):
            if bound is None:
                continue
            check_timescale(bound, name)


#: Name tokens that mark a log-scale value (``sfh_burst_log_tpeak_myr`` is
#: ``log10(Myr)``): the suffix names the unit inside the logarithm, so a linear
#: plausibility range does not apply.
_LOG_SCALE_TOKENS: frozenset[str] = frozenset({"log", "log10"})


def _timescale_unit(name: str) -> str | None:
    """Return the timescale suffix of *name*, or ``None`` if it has none.

    A log-scale name (one of :data:`_LOG_SCALE_TOKENS` among its ``_``-separated
    tokens) has no linear unit and returns ``None``.
    """
    if _LOG_SCALE_TOKENS.intersection(name.split("_")):
        return None
    for unit in TIMESCALE_BOUNDS:
        if name.endswith(unit):
            return unit
    return None
