# SPDX-License-Identifier: BSD-3-Clause
"""Catalog cell status: the one home of flux-unit conversion and per-cell status (#3547 D4).

A catalog cell (one galaxy in one band) is detected, an upper or lower limit, or absent
(``nodata``). Several entry points declare that status: a sentinel flux value
(``sentinels=`` in :func:`tengri.io.load_catalog`), CIGALE's ``lim_flag`` convention,
a censor column (``censor_cols`` of :class:`tengri.Catalog`), and the refusal of an
undeclared encoding. They all end in :func:`assign_cell_status`.

This module imports only numpy and :mod:`tengri.utils.conversions`, so both the catalog
reader (:mod:`tengri.io.catalog`) and the ingest (:mod:`tengri.inference.catalog_ingest`)
can import it without a cycle.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from tengri.utils.conversions import maggies_to_fnu, ujy_to_fnu

__all__ = [
    "CIGALE_INVALID_FLUX",
    "FLUX_UNITS",
    "LIMIT_STATUSES",
    "LIM_FLAGS",
    "CellStatus",
    "assign_cell_status",
    "convert_flux_to_cgs",
]

#: Flux units a catalog may be read in (the one unit system: the caller names it).
FLUX_UNITS = ("cgs_fnu", "mJy", "uJy", "maggies", "ab_mag")
#: CIGALE ``pdf_analysis`` ``lim_flag`` options (pcigale 2025.1).
LIM_FLAGS = ("none", "noscaling", "full")
#: Sentinel below which CIGALE's ``_check_invalid`` treats a flux as invalid (in the
#: table's own flux unit).
CIGALE_INVALID_FLUX = -9990.0
#: Limit statuses a sentinel or a flag may declare; the depth is the tuple's second item.
LIMIT_STATUSES = ("upper_limit", "lower_limit")


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


def assign_cell_status(
    flux,
    err,
    *,
    names,
    flux_unit,
    err_names=None,
    presence=None,
    sentinels=None,
    lim_flag=None,
    censor_flags=None,
    flag_names=None,
    refuse_undeclared=False,
) -> CellStatus:
    """Assign every cell's status from the declarations given, in the table's own units.

    This is the one place a cell's status is decided. Every entry point calls it:
    :func:`tengri.io.load_catalog` (``sentinels``) and ``ingest_catalog`` in
    :mod:`tengri.inference.catalog_ingest` (``lim_flag``, ``censor_cols`` as
    ``censor_flags``, and ``refuse_undeclared``).

    Parameters
    ----------
    flux, err : array_like, shape (N, n_bands)
        Raw flux and error in ``flux_unit``.
    names : tuple of str, length n_bands
        Flux column names, used in messages.
    flux_unit : str
        One of :data:`FLUX_UNITS`. A limit declared by ``sentinels`` in ``ab_mag`` is not
        yet supported (its depth is applied in flux space, not in magnitudes).
    err_names : tuple of str, optional
        Error column names for messages. Default ``"<name>_err"``.
    presence : array_like of bool, optional
        Cells already absent before this call (default: finite flux).
    sentinels : Mapping of float to status, optional
        A sentinel flux value (exact ``==``) and its status: ``"nodata"`` (the cell is
        absent, flux and error become NaN), ``("upper_limit", n_sigma)`` or
        ``("lower_limit", n_sigma)`` (the limit flux is ``n_sigma * err``, sigma is
        ``err``, censor is +1 or -1). A sentinel in an error column, or a limit cell whose
        error is not positive and finite, raises.
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
        ``refuse_undeclared`` is set, a sentinel in an error column, an invalid flag value,
        or a limit whose error is not positive.

    Notes
    -----
    ``sentinels``, ``lim_flag`` and ``censor_flags`` are mutually exclusive, except that
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

    if sentinels is not None and (lim_flag is not None or censor_flags is not None):
        raise ValueError(
            "sentinels, lim_flag and censor_cols declare the same cell status in three "
            "ways; give one of them."
        )
    if sentinels is None and lim_flag is None and censor_flags is None and refuse_undeclared:
        _refuse_undeclared(flux_arr, err_arr, names, err_names)

    censor = None
    if sentinels is not None:
        flux_arr, err_arr, present, censor = _apply_sentinels(
            flux_arr, err_arr, present, sentinels, names, err_names, flux_unit
        )
    if lim_flag is not None:
        if lim_flag not in LIM_FLAGS:
            raise ValueError(f"lim_flag={lim_flag!r} not in {list(LIM_FLAGS)}.")
        present, err_arr, censor = _apply_lim_flag(flux_arr, err_arr, present, lim_flag)
    if censor_flags is not None:
        censor = _checked_flags(censor_flags, flag_names or names, n_bands, flux_arr.shape[0])
    return CellStatus(flux_arr, err_arr, present, censor)


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
        "as noise. Declare the convention: pass sentinels= to load_catalog for sentinel "
        'flux values (nodata or limits), or lim_flag="full" for a CIGALE table (a sentinel '
        "flux is dropped and a negative error becomes an upper limit)."
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


def _parse_sentinel_status(key, status) -> tuple[str, float | None]:
    """Validate one declared sentinel status; return ``(kind, n_sigma)``.

    ``kind`` is ``"nodata"``, ``"upper_limit"`` or ``"lower_limit"``; ``n_sigma`` is
    ``None`` for ``"nodata"``. A bare limit name without a depth is refused.
    """
    allowed = "'nodata', ('upper_limit', n_sigma), ('lower_limit', n_sigma)"
    if isinstance(status, str):
        if status == "nodata":
            return "nodata", None
        if status in LIMIT_STATUSES:
            raise ValueError(
                f"sentinels[{key!r}] = {status!r} gives no depth. Declare it as "
                f"({status!r}, n_sigma), e.g. ({status!r}, 3.0): the limit flux is "
                f"n_sigma times the error."
            )
        raise ValueError(f"sentinels[{key!r}] = {status!r} is not a status; allowed: {allowed}.")
    if isinstance(status, tuple | list) and len(status) == 2 and status[0] in LIMIT_STATUSES:
        n_sigma = float(status[1])
        if not (np.isfinite(n_sigma) and n_sigma > 0.0):
            raise ValueError(
                f"sentinels[{key!r}] = {tuple(status)!r}: n_sigma must be finite and > 0; "
                f"got {status[1]!r}."
            )
        return str(status[0]), n_sigma
    raise ValueError(f"sentinels[{key!r}] = {status!r} is not a status; allowed: {allowed}.")


def _apply_sentinels(flux, err, present, sentinels, names, err_names, flux_unit):
    """Apply declared sentinel flux values to every band (exact ``==``); new arrays."""
    flux = flux.copy()
    err = err.copy()
    present = present.copy()
    censor = np.zeros(flux.shape, dtype=int)
    for key, status in sentinels.items():
        value = float(key)
        kind, n_sigma = _parse_sentinel_status(key, status)
        for j, band in enumerate(names):
            if np.any(err[:, j] == value):
                raise ValueError(
                    f"band {band!r}: declared sentinel {key!r} appears in the error column. "
                    f"A sentinel marks a flux cell; an error cannot be a sentinel."
                )
            hit = flux[:, j] == value
            if not hit.any():
                continue
            if kind == "nodata":
                flux[hit, j] = np.nan
                err[hit, j] = np.nan
                present[hit, j] = False
                continue
            if flux_unit == "ab_mag":
                raise ValueError(
                    f"band {band!r}: a {kind} sentinel in flux_unit='ab_mag' is not supported "
                    f"yet; the depth must be applied in flux space, not in magnitudes."
                )
            e = err[:, j]
            good = (e > 0.0) & np.isfinite(e)
            if not good[hit].all():
                rows = np.flatnonzero(hit & ~good).tolist()
                raise ValueError(
                    f"band {band!r}: {kind} sentinel {key!r} at row(s) {rows} has a "
                    f"non-positive or non-finite error; a limit needs a positive error as "
                    f"its sigma."
                )
            flux[hit, j] = n_sigma * e[hit]
            censor[hit, j] = 1 if kind == "upper_limit" else -1
    return flux, err, present, censor


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
