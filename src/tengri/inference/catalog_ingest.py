# SPDX-License-Identifier: BSD-3-Clause
"""Catalog ingestion: table -> contiguous, validated arrays.

#1317: pure, inference-free core for catalog-based model construction.
Handles name-matching, explicit units, NaN policies, and censor semantics.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from tengri.utils.conversions import maggies_to_fnu, ujy_to_fnu

__all__ = ["CatalogArrays", "ingest_catalog"]


class CatalogArrays(NamedTuple):
    """Contiguous validated arrays from table ingestion.

    Fields
    ------
    flux : ndarray, shape (N, n_bands)
        Spectral flux density [erg/s/cm²/Hz].
    noise : ndarray, shape (N, n_bands)
        Flux uncertainty [erg/s/cm²/Hz].
    redshift : ndarray, shape (N,) or None
        Redshift for each galaxy.
    censor : ndarray, shape (N, n_bands) or None
        Censoring flags: 0 (detected), 1 (upper limit), -1 (lower limit).
    presence : ndarray, shape (N, n_bands), dtype=bool
        Mask indicating which bands are present (not NaN-masked).
    n_galaxies : int
        Number of galaxies (N).
    band_names : tuple[str, ...]
        Band identifiers in order.
    """

    flux: np.ndarray
    noise: np.ndarray
    redshift: np.ndarray | None
    censor: np.ndarray | None
    presence: np.ndarray
    n_galaxies: int
    band_names: tuple[str, ...]


def ingest_catalog(
    table,
    *,
    photometry,
    flux_unit,
    flux_cols=None,
    err_cols=None,
    redshift_col=None,
    censor_cols=None,
    missing="error",
) -> CatalogArrays:
    """Convert a heterogeneous table into contiguous validated arrays.

    Parameters
    ----------
    table : dict-like or array-like
        Column mapping or object supporting `__getitem__[col] -> array`
        and `len()`. Covers dict, pandas DataFrame, astropy Table, etc.
    photometry : object
        Photometry object with `.names` and `.n_filters` attributes.
    flux_unit : str
        Unit of flux columns. One of: "cgs_fnu", "mJy", "uJy", "maggies",
        "ab_mag".
    flux_cols : list[str], optional
        Explicit flux column names. If None, use "{name}" for each band in
        photometry.
    err_cols : list[str], optional
        Explicit error column names. If None, use "{name}_err" for each band.
    redshift_col : str, optional
        Column name for redshifts. If None, redshift field is None.
    censor_cols : dict[str, str], optional
        Mapping from band name to censoring flag column. Flag values: 0
        (detected), 1 (upper limit), -1 (lower limit).
    missing : {"error", "mask"}, default "error"
        Policy for NaN flux values. "error" raises with guidance on
        `missing="mask"`; "mask" sets presence to False for that cell.

    Returns
    -------
    CatalogArrays

    Raises
    ------
    TypeError
        If flux_unit is not provided.
    ValueError
        If a required column is missing, flux/error counts mismatch,
        redshift is outside bounds, or a NaN is present with missing="error".

    Notes
    -----
    Unit conversions reuse `tengri.utils.conversions` where available.
    AB magnitude formula (Oke & Gunn 1983, ApJ 266, 713;
    DOI 10.1086/160817): f_ν = 10^(−0.4(m+48.60)) [erg/s/cm²/Hz];
    error propagation: σ_f = f · ln(10)/2.5 · σ_m.

    A NaN in an error column with finite flux is ALWAYS an error,
    regardless of missing= (an unknown uncertainty is not an absent band).
    """
    if flux_unit is None:
        raise TypeError("flux_unit is required")

    n_bands = photometry.n_filters
    band_names = tuple(photometry.names)

    # Step 1: Determine which columns to read
    if flux_cols is None:
        flux_cols = [f"{name}" for name in band_names]
    if err_cols is None:
        err_cols = [f"{name}_err" for name in band_names]

    # Validate counts
    if len(flux_cols) != n_bands:
        raise ValueError(f"flux_cols count ({len(flux_cols)}) != n_bands ({n_bands})")
    if len(err_cols) != n_bands:
        raise ValueError(f"err_cols count ({len(err_cols)}) != n_bands ({n_bands})")

    # Step 2: Extract arrays from table
    flux_arrays = []
    err_arrays = []
    try:
        for col_name in flux_cols:
            flux_arrays.append(np.asarray(table[col_name]))
    except (KeyError, TypeError) as e:
        # List actual columns for the user
        actual_cols = list(table.keys()) if hasattr(table, "keys") else "unknown"
        raise ValueError(f"Missing flux column '{col_name}'. Table columns: {actual_cols}") from e

    try:
        for col_name in err_cols:
            err_arrays.append(np.asarray(table[col_name]))
    except (KeyError, TypeError) as e:
        # List actual columns for the user
        actual_cols = list(table.keys()) if hasattr(table, "keys") else "unknown"
        raise ValueError(f"Missing error column '{col_name}'. Table columns: {actual_cols}") from e

    # Step 3: Stack into (N, n_bands) arrays
    flux_raw = np.column_stack(flux_arrays)  # (N, n_bands)
    err_raw = np.column_stack(err_arrays)  # (N, n_bands)
    n_galaxies = flux_raw.shape[0]

    # Step 4: Handle NaN in flux
    presence = np.isfinite(flux_raw)  # (N, n_bands), True = present
    if missing == "error":
        if not presence.all():
            i, j = np.where(~presence)
            raise ValueError(
                f"NaN flux detected at row(s) {i.tolist()}, band(s) {j.tolist()}. "
                f"Use missing='mask' to ignore absent bands, or convert sentinels "
                f"(e.g., -99) to NaN before ingestion."
            )
    elif missing == "mask":
        # Set NaN flux to finite placeholder (doesn't matter, presence=False)
        flux_raw = np.nan_to_num(flux_raw, nan=0.0)
    else:
        raise ValueError(f"missing='{missing}' not in {{'error', 'mask'}}")

    # Step 5: Check error NaN – ALWAYS an error (unknown uncertainty ≠ absent)
    err_has_nan = ~np.isfinite(err_raw)
    if err_has_nan.any():
        # But only raise if the flux is finite (absent flux doesn't need error)
        bad_mask = err_has_nan & presence
        if bad_mask.any():
            i, j = np.where(bad_mask)
            raise ValueError(
                f"NaN error with finite flux at row(s) {i.tolist()}, "
                f"band(s) {j.tolist()}. This is always an error; an unknown "
                f"uncertainty is not an absent band."
            )
    # Set error NaN to 0 (for absent bands where flux was also NaN)
    err_raw = np.nan_to_num(err_raw, nan=0.0)

    # Step 6: Convert units
    flux, err = _convert_flux_unit(flux_raw, err_raw, flux_unit)

    # Step 7: Read redshift if requested
    if redshift_col is not None:
        redshift = np.asarray(table[redshift_col])
        if redshift.ndim != 1 or len(redshift) != n_galaxies:
            raise ValueError(f"redshift must be 1D with length {n_galaxies}")
    else:
        redshift = None

    # Step 8: Read censoring if requested
    if censor_cols is not None:
        censor = np.zeros((n_galaxies, n_bands), dtype=int)
        for band_idx, band_name in enumerate(band_names):
            if band_name not in censor_cols:
                raise ValueError(f"Band '{band_name}' not in censor_cols mapping")
            col_name = censor_cols[band_name]
            censor[:, band_idx] = np.asarray(table[col_name])
    else:
        censor = None

    return CatalogArrays(
        flux=flux,
        noise=err,
        redshift=redshift,
        censor=censor,
        presence=presence,
        n_galaxies=n_galaxies,
        band_names=band_names,
    )


def _convert_flux_unit(
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
        Unit identifier.

    Returns
    -------
    flux, err : (ndarray, ndarray)
        Converted to CGS f_ν.

    Raises
    ------
    ValueError
        If flux_unit is unknown.
    """
    if flux_unit == "cgs_fnu":
        return flux_raw, err_raw
    elif flux_unit == "mJy":
        # mJy = 1e-3 Jy = 1e-3 * 1e-23 erg/s/cm²/Hz = 1e-26 erg/s/cm²/Hz
        factor = 1e-26
        return flux_raw * factor, err_raw * factor
    elif flux_unit == "uJy":
        # uJy: use existing converter
        return ujy_to_fnu(flux_raw), ujy_to_fnu(err_raw)
    elif flux_unit == "maggies":
        # maggies: use existing converter
        return maggies_to_fnu(flux_raw), maggies_to_fnu(err_raw)
    elif flux_unit == "ab_mag":
        # AB mag formula: f_ν = 10^(−0.4(m+48.60))
        # Error propagation: σ_f = f · ln(10)/2.5 · σ_m
        flux_cgs = 10.0 ** (-0.4 * (flux_raw + 48.60))
        err_cgs = flux_cgs * np.log(10.0) / 2.5 * err_raw
        return flux_cgs, err_cgs
    else:
        valid_units = {"cgs_fnu", "mJy", "uJy", "maggies", "ab_mag"}
        raise ValueError(f"Unknown flux_unit='{flux_unit}'. Valid options: {sorted(valid_units)}")
