# SPDX-License-Identifier: BSD-3-Clause
"""Catalog cell status: the one home of flux-unit conversion and per-cell status (#3547 D4).

A catalog cell (one galaxy in one band) is detected, an upper or lower limit, or absent
(``nodata``). Several entry points declare that status: sentinel flux values
(``sentinels=`` in :func:`tengri.io.load_catalog`), flag columns (``flag_cols=`` and
``flag_map=``), CIGALE's ``lim_flag`` convention, a censor column (``censor_cols`` of
:class:`tengri.Catalog`), and the refusal of an undeclared encoding. Each decodes its
declaration into one status per cell, and :func:`assign_cell_status` applies those.

This module imports only numpy and :mod:`tengri.utils.conversions`, so both the catalog
reader (:mod:`tengri.io.catalog`) and the ingest (:mod:`tengri.inference.catalog_ingest`)
can import it without a cycle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from tengri.utils.conversions import maggies_to_fnu, ujy_to_fnu

__all__ = [
    "CIGALE_INVALID_FLUX",
    "EMPTY_FLAG",
    "FLUX_UNITS",
    "LIMIT_STATUSES",
    "LIM_FLAGS",
    "CellStatus",
    "assign_cell_status",
    "convert_flux_to_cgs",
    "flag_specs",
    "normalize_sentinels",
    "sentinel_specs",
]

#: Flux units a catalog may be read in (the one unit system: the caller names it).
FLUX_UNITS = ("cgs_fnu", "mJy", "uJy", "maggies", "ab_mag")
#: CIGALE ``pdf_analysis`` ``lim_flag`` options (pcigale 2025.1).
LIM_FLAGS = ("none", "noscaling", "full")
#: Sentinel below which CIGALE's ``_check_invalid`` treats a flux as invalid (in the
#: table's own flux unit).
CIGALE_INVALID_FLUX = -9990.0
#: Limit statuses. A sentinel or a flag may give one as a tuple with its depth
#: ``(kind, n_sigma)``; a flag may also give the bare name, meaning the flux is the limit.
LIMIT_STATUSES = ("upper_limit", "lower_limit")
#: The flag key that matches an empty cell: an empty string after ``strip()``, or NaN/None.
EMPTY_FLAG = ""
_SENTINEL_DEFAULT = "*"


def convert_flux_to_cgs(
    flux_raw: np.ndarray,
    err_raw: np.ndarray,
    flux_unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flux and error to CGS f_ν [erg/s/cm²/Hz].

    Parameters
    ----------
    flux_raw : ndarray, shape (N, n_bands)
        Flux in the input unit.
    err_raw : ndarray, shape (N, n_bands)
        Error in the input unit.
    flux_unit : str
        Unit identifier, one of :data:`FLUX_UNITS`.

    Returns
    -------
    flux, err : (ndarray, ndarray)
        Converted to CGS f_ν.

    Raises
    ------
    ValueError
        If flux_unit is unknown.

    Notes
    -----
    ``ab_mag`` converts a magnitude m by f_ν = 10^(−0.4(m+48.60)) and its error by
    σ_f = f · ln(10)/2.5 · σ_m (Oke & Gunn 1983, ApJ 266, 713; DOI 10.1086/160817).
    """
    if flux_unit == "cgs_fnu":
        return flux_raw, err_raw
    elif flux_unit == "mJy":
        # mJy = 1e-3 Jy = 1e-3 * 1e-23 erg/s/cm²/Hz = 1e-26 erg/s/cm²/Hz
        factor = 1e-26
        return flux_raw * factor, err_raw * factor
    elif flux_unit == "uJy":
        return ujy_to_fnu(flux_raw), ujy_to_fnu(err_raw)
    elif flux_unit == "maggies":
        return maggies_to_fnu(flux_raw), maggies_to_fnu(err_raw)
    elif flux_unit == "ab_mag":
        flux_cgs = 10.0 ** (-0.4 * (flux_raw + 48.60))
        err_cgs = flux_cgs * np.log(10.0) / 2.5 * err_raw
        return flux_cgs, err_cgs
    else:
        raise ValueError(f"Unknown flux_unit='{flux_unit}'. Valid options: {list(FLUX_UNITS)}")


class CellStatus(NamedTuple):
    """Per-cell status arrays, in the table's own flux unit.

    Fields
    ------
    flux : ndarray, shape (N, n_bands)
        Flux; a limit cell holds its limit flux, an absent cell holds what was read.
    err : ndarray, shape (N, n_bands)
        1-sigma error (the sigma of a limit cell); absent cells are NaN or zero.
    presence : ndarray, shape (N, n_bands), dtype=bool
        False for an absent cell (``nodata`` or dropped by ``lim_flag``).
    censor : ndarray, shape (N, n_bands), dtype=int, or None
        0 detected, +1 upper limit, -1 lower limit; None when nothing declared a status.
    """

    flux: np.ndarray
    err: np.ndarray
    presence: np.ndarray
    censor: np.ndarray | None


# ── decoding: each declaration becomes one status per cell ──────────────────
#
# A status spec is one of:
#   None                      detected (the cell as read)
#   "nodata"                  absent: flux and error become NaN
#   "upper_limit"/"lower_limit"  bare limit: the cell's FLUX is the limit, its ERR the sigma
#   (kind, n_sigma)           tuple limit: the cell's ERR is the 1-sigma depth and the limit
#                             flux is n_sigma * err (the flux cell is ignored)


def _check_depth(where, status):
    """Return ``(kind, n_sigma)`` for a tuple limit; refuse a non-finite or non-positive depth."""
    kind, depth = status[0], status[1]
    n_sigma = float(depth)
    if not (math.isfinite(n_sigma) and n_sigma > 0.0):
        raise ValueError(
            f"{where} = {tuple(status)!r}: n_sigma must be finite and > 0; got {depth!r}."
        )
    return kind, n_sigma


def sentinel_specs(flux, err, mapping, band):
    """Decode the sentinel flux values of one band into a status spec per row.

    Parameters
    ----------
    flux, err : array_like, shape (N,)
        Raw flux and error of the band, in the file's unit.
    mapping : Mapping of float to status, or None
        A sentinel value (matched with exact ``==``) and its status: ``"nodata"`` or
        ``("upper_limit", n_sigma)`` / ``("lower_limit", n_sigma)``. A bare limit name
        is refused: a sentinel's limit needs its depth.
    band : str
        Band name, used in messages.

    Returns
    -------
    ndarray, shape (N,), dtype=object
        The status spec of each row (see the module notes); None for a detected row.

    Raises
    ------
    ValueError
        For a bare limit or an unknown status, or when a declared sentinel appears in the
        error column.
    """
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    specs = np.full(flux.shape, None, dtype=object)
    for key, status in (mapping or {}).items():
        value = float(key)
        if np.any(err == value):
            raise ValueError(
                f"band {band!r}: declared sentinel {key!r} appears in the error column. "
                f"A sentinel marks a flux cell; an error cannot be a sentinel."
            )
        hit = flux == value
        if isinstance(status, str):
            if status == "nodata":
                specs[hit] = "nodata"
                continue
            if status in LIMIT_STATUSES:
                raise ValueError(
                    f"sentinels[{key!r}] = {status!r} gives no depth. Declare it as "
                    f"({status!r}, n_sigma), e.g. ({status!r}, 3.0): the limit flux is "
                    f"n_sigma times the error."
                )
            raise ValueError(
                f"sentinels[{key!r}] = {status!r} is not a status; allowed: "
                f"'nodata', ('upper_limit', n_sigma), ('lower_limit', n_sigma)."
            )
        if isinstance(status, tuple | list) and len(status) == 2 and status[0] in LIMIT_STATUSES:
            spec = _check_depth(f"sentinels[{key!r}]", status)
            for i in np.flatnonzero(hit):  # a tuple cannot be broadcast into a masked slice
                specs[i] = spec
            continue
        raise ValueError(
            f"sentinels[{key!r}] = {status!r} is not a status; allowed: "
            f"'nodata', ('upper_limit', n_sigma), ('lower_limit', n_sigma)."
        )
    return specs


def normalize_sentinels(sentinels, names):
    """Return the sentinel mapping of each band: the flat form, or ``{"*": ..., band: ...}``.

    The flat form (values are statuses) applies to every band, the same as ``"*"``. The
    nested form (values are mappings) gives a default under ``"*"``; a band key overrides it
    value by value. The two forms cannot be mixed. A flux column named ``"*"`` is refused,
    because ``"*"`` is the default key.

    Parameters
    ----------
    sentinels : Mapping or None
        The declaration, as above.
    names : sequence of str
        The band (flux column) names of the table.

    Returns
    -------
    dict of str to dict, or None
        For each band, the flat mapping of sentinel values to statuses that applies to it.
    """
    if sentinels is None:
        return None
    names = tuple(names)
    if _SENTINEL_DEFAULT in names:
        raise ValueError(
            f'a flux column named "{_SENTINEL_DEFAULT}" is not allowed: "{_SENTINEL_DEFAULT}" is '
            f"the default key of a nested sentinels mapping. Rename the column."
        )
    values = list(sentinels.values())
    nested = [isinstance(v, Mapping) for v in values]
    if values and any(nested) and not all(nested):
        raise ValueError(
            "sentinels mixes statuses with per-band mappings. Use the flat form (statuses), "
            f'or the nested form (each value a mapping, with "{_SENTINEL_DEFAULT}" for every '
            f"band)."
        )
    if not (values and all(nested)):
        return {band: dict(sentinels) for band in names}
    default = dict(sentinels.get(_SENTINEL_DEFAULT, {}))
    per_band = {k: v for k, v in sentinels.items() if k != _SENTINEL_DEFAULT}
    unknown = sorted(set(per_band) - set(names))
    if unknown:
        raise ValueError(f"sentinels names band(s) {unknown} that the table does not have.")
    return {band: {**default, **dict(per_band.get(band, {}))} for band in names}


def _flag_key(value):
    """Normalize one flag cell or flag-map key; see :data:`EMPTY_FLAG`.

    None and NaN (and any string that is empty after ``strip()``) map to :data:`EMPTY_FLAG`.
    Numbers, and strings that read as a number, map to a float; other strings map to their
    stripped text. Numeric flags match by ``==``.
    """
    if value is None:
        return EMPTY_FLAG
    if isinstance(value, bool | np.bool_):
        return float(value)
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return EMPTY_FLAG if math.isnan(number) else number
    text = str(value).strip()
    if text == "":
        return EMPTY_FLAG
    try:
        number = float(text)
    except ValueError:
        return text
    return EMPTY_FLAG if math.isnan(number) else number


def _parse_flag_status(key, status):
    """Validate one ``flag_map`` status; return the spec the applier reads."""
    if isinstance(status, str):
        if status in ("detected", "nodata") or status in LIMIT_STATUSES:
            return None if status == "detected" else status
    elif isinstance(status, tuple | list) and len(status) == 2 and status[0] in LIMIT_STATUSES:
        return _check_depth(f"flag_map[{key!r}]", status)
    raise ValueError(
        f"flag_map[{key!r}] = {status!r} is not a status; allowed: 'detected', 'nodata', "
        f"'upper_limit', 'lower_limit', or ('upper_limit' | 'lower_limit', n_sigma)."
    )


def flag_specs(values, flag_map, column):
    """Decode one flag column into a status spec per row, through ``flag_map``.

    Parameters
    ----------
    values : array_like, shape (N,)
        The raw flag cells. Strings are compared after ``strip()``; an empty or NaN cell
        matches the explicit key ``""``; numbers match by ``==``.
    flag_map : Mapping
        Flag value to status: ``"detected"``, ``"nodata"``, a bare limit name (the flux is
        the limit, the error its sigma), or ``("upper_limit"|"lower_limit", n_sigma)`` (the
        error is the depth, the flux is ignored).
    column : str
        The flag column name, used in messages.

    Returns
    -------
    ndarray, shape (N,), dtype=object
        The status spec of each row (see the module notes).

    Raises
    ------
    ValueError
        For an unmapped flag value (listing the values and the first rows) or an invalid
        status in ``flag_map``.
    """
    table = {}
    for key, status in flag_map.items():
        norm = _flag_key(key)
        if norm in table:
            raise ValueError(f"flag_map has two keys that match the same flag value: {key!r}.")
        table[norm] = _parse_flag_status(key, status)
    values = list(values)
    specs = np.empty(len(values), dtype=object)
    unmapped_rows, unmapped_values = [], []
    for i, raw in enumerate(values):
        norm = _flag_key(raw)
        if norm in table:
            specs[i] = table[norm]
        else:
            unmapped_rows.append(i)
            if raw not in unmapped_values:
                unmapped_values.append(raw)
    if unmapped_rows:
        raise ValueError(
            f"flag column {column!r}: value(s) {unmapped_values} have no entry in flag_map; "
            f'first row(s) {unmapped_rows[:5]}. Map every flag value (use "" for an empty cell).'
        )
    return specs


# ── the one applier ─────────────────────────────────────────────────────────


class _Cell(NamedTuple):
    kind: str
    depth: float | None


def _split(spec) -> _Cell:
    """Read a status spec as ``(kind, depth)``: depth is set only for a tuple limit."""
    if spec is None or spec == "detected":
        return _Cell("detected", None)
    if isinstance(spec, tuple):
        return _Cell(spec[0], spec[1])
    return _Cell(spec, None)


def assign_cell_status(
    flux,
    err,
    *,
    names,
    flux_unit,
    err_names=None,
    presence=None,
    cell_status=None,
    lim_flag=None,
    censor_flags=None,
    flag_names=None,
    refuse_undeclared=False,
) -> CellStatus:
    """Assign every cell's status from the declarations given, in the table's own units.

    This is the one place a cell's status is applied. Every entry point decodes its
    declaration into a status spec per cell and calls it: :func:`tengri.io.load_catalog`
    (``sentinels``, ``flag_cols``) and ``ingest_catalog`` in
    :mod:`tengri.inference.catalog_ingest` (``lim_flag``, ``censor_cols`` as
    ``censor_flags``, and ``refuse_undeclared``).

    Parameters
    ----------
    flux, err : array_like, shape (N, n_bands)
        Raw flux and error in ``flux_unit``.
    names : tuple of str, length n_bands
        Flux column names, used in messages.
    flux_unit : str
        One of :data:`FLUX_UNITS`. A limit in ``ab_mag`` is not supported yet: its depth
        must be applied in flux space, and that conversion is not written here.
    err_names : tuple of str, optional
        Error column names for messages. Default ``"<name>_err"``.
    presence : array_like of bool, optional
        Cells already absent before this call (default: finite flux).
    cell_status : sequence of ndarray, optional
        One object array of status specs per band, each of length N (see
        :func:`sentinel_specs` and :func:`flag_specs`); None entries are detected.
    lim_flag : {"none", "noscaling", "full"}, optional
        CIGALE's ``lim_flag`` rule (pcigale ``_check_invalid``). ``"none"``: a cell with
        ``err <= 0`` or a flux below -9990 is dropped. ``"noscaling"`` / ``"full"``: a
        cell with ``err < 0`` is an upper limit at its flux with sigma ``|err|``; ``err == 0``
        or a flux below -9990 is dropped.
    censor_flags : sequence of array_like, optional
        One flag array per band (0, 1 or -1; a boolean array is refused). Used as the
        censor column of the cells.
    flag_names : tuple of str, optional
        Censor column names for messages.
    refuse_undeclared : bool, default False
        When True and no declaration was given, a CIGALE sentinel flux (below -9990) or a
        negative error raises, naming the column and row count and both declarations.

    Returns
    -------
    CellStatus

    Raises
    ------
    ValueError
        For an invalid or conflicting declaration, an undeclared encoding when
        ``refuse_undeclared`` is set, a limit in ``ab_mag``, or a limit cell whose error is
        not positive and finite (the row count and first rows are named).

    Notes
    -----
    ``cell_status``, ``lim_flag`` and ``censor_flags`` are mutually exclusive, except that
    ``censor_flags`` may accompany ``lim_flag="none"`` (the flags then take precedence for
    the censor column, as in CIGALE's table reader). Not JIT-traced: runs once on host
    arrays during the read or the ingest.
    """
    flux_arr = np.array(flux, dtype=float)
    err_arr = np.array(err, dtype=float)
    n_bands = flux_arr.shape[1]
    names = tuple(names)
    err_names = tuple(err_names) if err_names is not None else tuple(f"{n}_err" for n in names)
    if len(names) != n_bands or len(err_names) != n_bands:
        raise ValueError(f"names/err_names must name the {n_bands} band column(s)")
    present = np.isfinite(flux_arr) if presence is None else np.array(presence, dtype=bool)

    if cell_status is not None and (lim_flag is not None or censor_flags is not None):
        raise ValueError(
            "a per-cell status (sentinels, flag columns), lim_flag and censor_cols declare "
            "the same cell status in different ways; give one of them."
        )
    if cell_status is None and lim_flag is None and censor_flags is None and refuse_undeclared:
        _refuse_undeclared(flux_arr, err_arr, names, err_names)

    censor = None
    if cell_status is not None:
        flux_arr, err_arr, present, censor = _apply_cell_status(
            flux_arr, err_arr, present, cell_status, names, flux_unit
        )
    if lim_flag is not None:
        if lim_flag not in LIM_FLAGS:
            raise ValueError(f"lim_flag={lim_flag!r} not in {list(LIM_FLAGS)}.")
        present, err_arr, censor = _apply_lim_flag(flux_arr, err_arr, present, lim_flag)
    if censor_flags is not None:
        censor = _checked_flags(censor_flags, flag_names or names, n_bands, flux_arr.shape[0])
    return CellStatus(flux_arr, err_arr, present, censor)


def _apply_cell_status(flux, err, present, cell_status, names, flux_unit):
    """Apply one status spec per cell (the one applier); return new arrays and the censor."""
    flux = flux.copy()
    err = err.copy()
    present = present.copy()
    censor = np.zeros(flux.shape, dtype=int)
    for j, band in enumerate(names):
        specs = cell_status[j]
        if specs is None:
            continue
        bad_rows = []
        for i, spec in enumerate(specs):
            cell = _split(spec)
            if cell.kind == "detected":
                continue
            if cell.kind == "nodata":
                flux[i, j] = np.nan
                err[i, j] = np.nan
                present[i, j] = False
                continue
            if cell.kind not in LIMIT_STATUSES:
                raise ValueError(f"band {band!r}: unknown status {cell.kind!r} at row {i}.")
            if flux_unit == "ab_mag":
                raise ValueError(
                    f"band {band!r}: a {cell.kind} limit in flux_unit='ab_mag' is not "
                    f"supported yet; the depth must be applied in flux space, not in "
                    f"magnitudes."
                )
            sigma = err[i, j]
            if not (sigma > 0.0 and math.isfinite(sigma)):
                bad_rows.append(i)
                continue
            if cell.depth is not None:
                flux[i, j] = cell.depth * sigma
            censor[i, j] = 1 if cell.kind == "upper_limit" else -1
        if bad_rows:
            raise ValueError(
                f"band {band!r}: {len(bad_rows)} row(s) {bad_rows[:5]} declared as a limit with "
                f"a non-positive or non-finite error; a limit needs a positive error as its sigma."
            )
    return flux, err, present, censor


def _describe_offending(kind, offending, columns):
    """Name each column holding an offending cell and how many rows it has."""
    described = []
    for j, name in enumerate(columns):
        n_rows = int(offending[:, j].sum())
        if n_rows:
            plural = "row" if n_rows == 1 else "rows"
            described.append(f"{kind} in column {name!r}: {n_rows} {plural}")
    return "; ".join(described)


def _refuse_undeclared(flux, err, names, err_names):
    """Refuse a CIGALE sentinel flux or negative error that no declaration covers (#3547)."""
    # A non-finite flux is an absent band (the missing= policy), not a sentinel value.
    sentinel = np.isfinite(flux) & (flux < CIGALE_INVALID_FLUX)
    negative = np.isfinite(err) & (err < 0.0)
    if not (sentinel.any() or negative.any()):
        return
    parts = []
    if sentinel.any():
        parts.append(
            _describe_offending(f"sentinel flux (below {CIGALE_INVALID_FLUX:g})", sentinel, names)
        )
    if negative.any():
        parts.append(_describe_offending("negative error", negative, err_names))
    raise ValueError(
        "; ".join(parts) + ". Without a declaration these would be read as a detection or "
        "as noise. Declare the convention: pass sentinels= or flag_cols= to load_catalog, or "
        'lim_flag="full" for a CIGALE table (a sentinel flux is dropped and a negative error '
        "becomes an upper limit)."
    )


def _apply_lim_flag(flux, err, present, lim_flag):
    """CIGALE ``lim_flag`` rule (pcigale ``_check_invalid``); returns new arrays."""
    invalid_flux = flux < CIGALE_INVALID_FLUX
    if lim_flag == "none":
        drop = invalid_flux | (err <= 0.0)
        limit = np.zeros(drop.shape, dtype=bool)
    else:
        drop = invalid_flux | (err == 0.0)
        limit = (~drop) & (err < 0.0)
    new_present = present & ~drop
    new_err = np.where(limit, -err, np.where(drop, 0.0, err))
    censor = None if lim_flag == "none" else limit.astype(int)
    return new_present, new_err, censor


def _checked_flags(flags, names, n_bands, n_rows):
    """Validate censor flag columns (0, 1, -1; never a boolean mask) and stack them."""
    if len(flags) != n_bands:
        raise ValueError(f"censor flags: {len(flags)} column(s) for {n_bands} band(s)")
    columns = []
    for name, raw in zip(names, flags, strict=True):
        col = np.asarray(raw)
        # Reject boolean flags: a bool ``True`` would silently cast to 1
        # ("upper limit"), laundering an intended include-mask past the Fitter's
        # data_mask guard (spec §3.3). Flags are integers {0,1,-1}.
        if col.dtype == bool:
            raise ValueError(
                f"censor column '{name}' is boolean; censor flags are integers 0 (detected), "
                f"1 (upper limit), -1 (lower limit). A boolean mask is not a censor flag, "
                f"convert it explicitly."
            )
        # Reject garbage flag values (-99, 2, 0.5, NaN, ...) at the seam.
        in_range = np.isin(col, (-1, 0, 1))
        if not in_range.all():
            bad = np.unique(col[~in_range]).tolist()
            raise ValueError(
                f"censor column '{name}' has invalid flag value(s) {bad}; "
                f"allowed: 0 (detected), 1 (upper limit), -1 (lower limit)."
            )
        columns.append(col.astype(int))
    censor = np.column_stack(columns)
    if censor.shape != (n_rows, n_bands):
        raise ValueError(f"censor flags have shape {censor.shape}, expected {(n_rows, n_bands)}")
    return censor
