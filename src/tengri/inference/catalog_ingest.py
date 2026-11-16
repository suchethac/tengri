# SPDX-License-Identifier: BSD-3-Clause
"""Catalog ingestion: table -> contiguous, validated arrays.

#1317: pure, inference-free core for catalog-based model construction.
Handles name-matching, explicit units, NaN policies, and censor semantics.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from tengri.utils.catalog_status import (
    LIM_FLAGS,
    assign_cell_status,
    convert_flux_to_cgs,
)

__all__ = ["CatalogArrays"]


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
    line_flux_obs : ndarray, shape (N, n_lines) or None
        Per-galaxy observed emission-line fluxes [erg/s/cm²].
    line_flux_err : ndarray, shape (N, n_lines) or None
        Per-galaxy emission-line flux uncertainties [erg/s/cm²].
    line_censor : ndarray, shape (N, n_lines) or None
        Per-galaxy emission-line censoring flags: 0 (detected), 1 (upper
        limit), -1 (lower limit). Same convention as ``censor``, which is
        the photometric band axis.
    """

    flux: np.ndarray
    noise: np.ndarray
    redshift: np.ndarray | None
    censor: np.ndarray | None
    presence: np.ndarray
    n_galaxies: int
    band_names: tuple[str, ...]
    line_flux_obs: np.ndarray | None = None
    line_flux_err: np.ndarray | None = None
    line_censor: np.ndarray | None = None


def ingest_catalog(
    table,
    *,
    photometry,
    flux_unit,
    flux_cols=None,
    err_cols=None,
    redshift_col=None,
    censor_cols=None,
    line_cols=None,
    line_err_cols=None,
    line_censor_cols=None,
    missing="error",
    default_relative_error=None,
    lim_flag=None,
    observed_relative_error=None,
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
        Flux column names **in your table**, they need not resemble the band
        names. If None, defaults to ``"{name}"`` for each band in photometry.

        Binding is **positional**: ``flux_cols[i]`` is read as the flux for
        ``photometry.names[i]``. That is the only possible rule once the names
        are arbitrary, and it is why the length must equal the band count. So
        order these to match the observation, not the table's own layout::

            # bands are ("sdss_g", "sdss_r")
            flux_cols = ["FLUX_G", "FLUX_R"]  # g -> FLUX_G, r -> FLUX_R
            flux_cols = ["FLUX_R", "FLUX_G"]  # silently swaps the two bands

        A name absent from the table raises, listing the table's actual
        columns.
    err_cols : list[str], optional
        Error column names in your table, bound positionally in the same way.
        If None, defaults to ``"{name}_err"`` for each band.
    redshift_col : str, optional
        Column name for redshifts. If None, redshift field is None.
    censor_cols : dict[str, str], optional
        Mapping from band name to censoring flag column. Flag values: 0
        (detected), 1 (upper limit), -1 (lower limit).
    line_cols : list[str], optional
        Emission-line flux column names in your table, bound positionally to the
        observation's line order. If None and the observation carries line fluxes,
        raises ValueError with guidance. If None and no line fluxes are configured,
        this parameter is ignored.
    line_err_cols : list[str], optional
        Emission-line error column names, bound positionally the same way.
        If None, defaults to ``"{name}_err"`` for each line, matching the
        ``err_cols`` convention.
    line_censor_cols : list[str], optional
        Emission-line censoring-flag column names, bound positionally the
        same way. Flag values: 0 (detected), 1 (upper limit), -1 (lower
        limit) -- the ``censor_cols`` convention, on the line axis instead
        of the band axis. A list rather than a name-keyed dict because its
        two siblings ``line_cols`` / ``line_err_cols`` bind positionally.
    missing : {"error", "mask"}, default "error"
        Policy for NaN flux values. "error" raises with guidance on
        `missing="mask"`; "mask" sets presence to False for that cell.

    default_relative_error : float, optional
        CIGALE's ``defaulterror`` (Boquien et al. 2019, A&A 622, A103). When given,
        a flux column whose error column is absent from the table gets
        ``error = default_relative_error * |flux|`` (a warning names the band),
        instead of raising. ``None`` (default) keeps the refusal. Must be >= 0.
        Not available with ``flux_unit="ab_mag"`` (a magnitude error is not a
        fraction of the flux).

        A table with CIGALE's sentinels (a flux below -9990) needs ``lim_flag``
        together with this option: CIGALE always follows the default error with its
        invalid-data check, which ``lim_flag`` reproduces. Without ``lim_flag`` the
        sentinel is refused, not given the error ``default_relative_error * |flux|``.
    lim_flag : {"none", "noscaling", "full"}, optional
        CIGALE's ``lim_flag`` convention (Boquien et al. 2019, A&A 622, A103) for
        a table whose error column encodes limits (pcigale
        ``managers/observations.py`` ``_check_invalid``). ``"none"``: a band
        with ``err <= 0`` (or a flux below -9990) is dropped (``presence``
        False). ``"noscaling"`` and ``"full"``: a band with ``err < 0`` is an
        upper limit at the flux value with sigma = ``|err|`` (``censor`` = 1,
        ``noise`` = ``|err|``); ``err == 0`` or a flux below -9990 is dropped.

        The -9990 threshold is compared with the table's own numbers, in
        ``flux_unit`` (CIGALE tables are in mJy): a table in another unit with a
        different sentinel (for example -99) is not caught by it.

        ``"noscaling"`` and ``"full"`` give the same arrays here: CIGALE's
        ``"full"`` adds an analytic rescaling of the model amplitude that tengri
        does not perform (the amplitude is a fitted parameter). ``None`` (default)
        refuses a table that carries either encoding: a flux below -9990 or a
        negative error raises ``ValueError`` naming the column and the row count,
        and asks for ``lim_flag="full"`` (the declaration of the CIGALE convention).
        Otherwise errors are untouched. The two limit-producing options refuse
        ``censor_cols`` (two sources of censor flags), and every option refuses
        ``flux_unit="ab_mag"``.

    observed_relative_error : float, optional
        CIGALE's ``additionalerror`` (Boquien et al. 2019, A&A 622, A103), scaled by
        the observed flux. Each detected band (``presence`` true, not a limit, error
        >= 0) gets ``noise = sqrt(noise**2 + (observed_relative_error * |flux|)**2)``
        (pcigale ``managers/observations.py`` ``_add_model_error``). Limits and absent
        bands are untouched. ``None`` (default) adds nothing. Must be finite and >= 0.
        Not available with ``flux_unit="ab_mag"``.

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
    _validate_cigale_input_options(default_relative_error, lim_flag, flux_unit, censor_cols)
    _validate_observed_floor(observed_relative_error, flux_unit)

    n_bands = photometry.n_filters
    band_names = tuple(photometry.names)

    # Step 1: Determine which columns to read
    if flux_cols is None:
        flux_cols = [f"{name}" for name in band_names]
    else:
        # No band-name check here, deliberately. This used to require every
        # entry to already BE a band name, which meant `flux_cols` could only
        # ever be a permutation of its own default; it could not name a real
        # catalog column, the only reason the parameter exists (#1458). Column
        # existence is validated against the TABLE in step 2 below, which is
        # the right referent and already reports the actual column list.
        # `err_cols` never had the band-name check, and that asymmetry is what
        # showed the check was a mistake rather than a contract.
        flux_cols = list(flux_cols)

    if err_cols is None:
        err_cols = [f"{name}_err" for name in band_names]
    else:
        err_cols = list(err_cols)

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

    for band_name, col_name, flux_arr in zip(band_names, err_cols, flux_arrays, strict=True):
        try:
            err_arrays.append(np.asarray(table[col_name]))
        except (KeyError, TypeError) as e:
            if default_relative_error is None:
                # List actual columns for the user
                actual_cols = list(table.keys()) if hasattr(table, "keys") else "unknown"
                raise ValueError(
                    f"Missing error column '{col_name}'. Table columns: {actual_cols}"
                ) from e
            # CIGALE ``defaulterror``: error = fraction * |flux| (pcigale
            # ``ObservationsManagerPassbands._check_errors``).
            warnings.warn(
                f"{default_relative_error * 100}% of {band_name} taken as errors "
                f"(column '{col_name}' absent; default_relative_error).",
                UserWarning,
                stacklevel=2,
            )
            err_arrays.append(np.abs(np.asarray(flux_arr, dtype=float)) * default_relative_error)

    # Step 3: Stack into (N, n_bands) arrays
    flux_raw = np.column_stack(flux_arrays)  # (N, n_bands)
    err_raw = np.column_stack(err_arrays)  # (N, n_bands)
    n_galaxies = flux_raw.shape[0]
    if lim_flag is None:
        # Refuses an undeclared sentinel flux or negative error (#3547 D4).
        assign_cell_status(
            flux_raw,
            err_raw,
            names=flux_cols,
            err_names=err_cols,
            flux_unit=flux_unit,
            refuse_undeclared=True,
        )

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
        # Absent placeholder (presence=False): NaN and +/-inf alike. nan_to_num would
        # turn +inf into 1.8e308, a finite value the likelihood then reads.
        flux_raw = _zero_non_finite(flux_raw)
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
    # Set non-finite error to 0 (absent bands: their flux was NaN or inf)
    err_raw = _zero_non_finite(err_raw)

    # Step 5b: CIGALE ``lim_flag`` encoding of limits / invalid bands
    censor_from_flag = None
    if lim_flag is not None:
        status = assign_cell_status(
            flux_raw,
            err_raw,
            names=flux_cols,
            err_names=err_cols,
            flux_unit=flux_unit,
            presence=presence,
            lim_flag=lim_flag,
        )
        presence, err_raw, censor_from_flag = status.presence, status.err, status.censor

    # Step 6: Convert units
    flux, err = convert_flux_to_cgs(flux_raw, err_raw, flux_unit)

    # Step 7: Read redshift if requested
    if redshift_col is not None:
        redshift = np.asarray(table[redshift_col], dtype=float)
        if redshift.ndim != 1 or len(redshift) != n_galaxies:
            raise ValueError(f"redshift must be 1D with length {n_galaxies}")
        # A non-finite or negative redshift produces a NaN/garbage loss with no
        # trail back to the offending row, reject it at this seam.
        if not np.all(np.isfinite(redshift)):
            bad = np.where(~np.isfinite(redshift))[0].tolist()
            raise ValueError(
                f"redshift has non-finite value(s) at row(s) {bad}. A NaN/inf "
                f"redshift becomes a NaN loss with no trail to the row; convert "
                f"sentinels (e.g. -99) and fix missing redshifts before ingestion."
            )
        if np.any(redshift < 0.0):
            bad = np.where(redshift < 0.0)[0].tolist()
            raise ValueError(
                f"redshift has negative value(s) at row(s) {bad}; redshift must be >= 0."
            )
    else:
        redshift = None

    # Step 8: Read censoring if requested
    if censor_cols is not None:
        flag_names, flag_columns = [], []
        for band_name in band_names:
            if band_name not in censor_cols:
                raise ValueError(f"Band '{band_name}' not in censor_cols mapping")
            flag_names.append(censor_cols[band_name])
            flag_columns.append(np.asarray(table[censor_cols[band_name]]))
        censor = assign_cell_status(
            flux_raw,
            err_raw,
            names=band_names,
            flux_unit=flux_unit,
            censor_flags=flag_columns,
            flag_names=flag_names,
        ).censor
    elif censor_from_flag is not None:
        censor = censor_from_flag
    else:
        censor = None

    # Step 8b: CIGALE ``additionalerror`` on the detected bands (opt-in)
    err = _apply_observed_floor(flux, err, presence, censor, observed_relative_error)

    # Step 9: Read emission-line fluxes if requested
    line_flux_obs = None
    line_flux_err = None
    line_censor = None
    if line_cols is None and line_censor_cols is not None:
        raise ValueError(
            "line_censor_cols was given without line_cols. A censor flag "
            "marks a line as a limit rather than a detection, so it needs "
            "the line it refers to. Pass line_cols=[...] as well."
        )
    if line_cols is not None:
        # User explicitly provided line columns
        line_cols = list(line_cols)
        if line_err_cols is None:
            # Infer error column names from line names
            # We need the line names from the table structure, but we don't have
            # them here. Fall back to assuming "{col}_err" pattern for each.
            line_err_cols = [f"{col}_err" for col in line_cols]
        else:
            line_err_cols = list(line_err_cols)

        # Validate counts match
        if len(line_cols) != len(line_err_cols):
            raise ValueError(
                f"line_cols count ({len(line_cols)}) != line_err_cols count ({len(line_err_cols)})"
            )

        # Extract line flux arrays
        line_flux_arrays = []
        try:
            for col_name in line_cols:
                line_flux_arrays.append(np.asarray(table[col_name]))
        except (KeyError, TypeError) as e:
            actual_cols = list(table.keys()) if hasattr(table, "keys") else "unknown"
            raise ValueError(
                f"Missing line flux column '{col_name}'. Table columns: {actual_cols}"
            ) from e

        line_err_arrays = []
        try:
            for col_name in line_err_cols:
                line_err_arrays.append(np.asarray(table[col_name]))
        except (KeyError, TypeError) as e:
            actual_cols = list(table.keys()) if hasattr(table, "keys") else "unknown"
            raise ValueError(
                f"Missing line error column '{col_name}'. Table columns: {actual_cols}"
            ) from e

        # Stack into (N, n_lines) arrays
        line_flux_obs = np.column_stack(line_flux_arrays)
        line_flux_err = np.column_stack(line_err_arrays)

        # Per-galaxy line censoring (#1469). Same flag convention and the
        # same boolean refusal as the photometric ``censor_cols`` above: a
        # boolean column is an include-mask, and ``astype(int)`` would turn
        # every True into "upper limit" without saying so.
        if line_censor_cols is not None:
            line_censor_cols = list(line_censor_cols)
            if len(line_censor_cols) != len(line_cols):
                raise ValueError(
                    f"line_censor_cols count ({len(line_censor_cols)}) != "
                    f"line_cols count ({len(line_cols)}); one censor column "
                    "per line column, bound in the same order."
                )
            censor_arrays = []
            for col_name in line_censor_cols:
                try:
                    col = np.asarray(table[col_name])
                except (KeyError, TypeError) as e:
                    actual_cols = list(table.keys()) if hasattr(table, "keys") else "unknown"
                    raise ValueError(
                        f"Missing line censor column '{col_name}'. Table columns: {actual_cols}"
                    ) from e
                if col.dtype == bool:
                    raise ValueError(
                        f"line censor column '{col_name}' is boolean; censor "
                        "flags are integers 0 (detected), 1 (upper limit), "
                        "-1 (lower limit). A boolean mask is not a censor "
                        "flag, convert it explicitly."
                    )
                bad = sorted(set(np.unique(col).tolist()) - {0, 1, -1})
                if bad:
                    raise ValueError(
                        f"line censor column '{col_name}' has invalid flag "
                        f"value(s) {bad}; allowed: 0 (detected), 1 (upper "
                        "limit), -1 (lower limit)."
                    )
                censor_arrays.append(col.astype(int))
            line_censor = np.column_stack(censor_arrays)
            if line_censor.shape[0] != n_galaxies:
                raise ValueError(
                    f"line censor array has {line_censor.shape[0]} rows, "
                    f"but flux has {n_galaxies} rows"
                )

        # Validate shapes
        if line_flux_obs.shape[0] != n_galaxies:
            raise ValueError(
                f"line flux array has {line_flux_obs.shape[0]} rows, "
                f"but flux has {n_galaxies} rows"
            )
        if line_flux_err.shape[0] != n_galaxies:
            raise ValueError(
                f"line error array has {line_flux_err.shape[0]} rows, "
                f"but flux has {n_galaxies} rows"
            )

        # Handle NaN in line fluxes (same policy as photometric flux)
        line_presence = np.isfinite(line_flux_obs)
        if missing == "error":
            if not line_presence.all():
                i, j = np.where(~line_presence)
                raise ValueError(
                    f"NaN line flux detected at row(s) {i.tolist()}, "
                    f"line(s) {j.tolist()}. Use missing='mask' to ignore absent lines, "
                    f"or convert sentinels before ingestion."
                )
        elif missing == "mask":
            # Set NaN line flux to finite placeholder
            line_flux_obs = _zero_non_finite(line_flux_obs)

        # Check error NaN – ALWAYS an error
        line_err_has_nan = ~np.isfinite(line_flux_err)
        if line_err_has_nan.any():
            bad_mask = line_err_has_nan & line_presence
            if bad_mask.any():
                i, j = np.where(bad_mask)
                raise ValueError(
                    f"NaN line error with finite flux at row(s) {i.tolist()}, "
                    f"line(s) {j.tolist()}. This is always an error; an unknown "
                    f"uncertainty is not an absent line."
                )
        line_flux_err = _zero_non_finite(line_flux_err)

    return CatalogArrays(
        flux=flux,
        noise=err,
        redshift=redshift,
        censor=censor,
        presence=presence,
        n_galaxies=n_galaxies,
        band_names=band_names,
        line_flux_obs=line_flux_obs,
        line_flux_err=line_flux_err,
        line_censor=line_censor,
    )


def _validate_observed_floor(observed_relative_error, flux_unit):
    """Refuse an invalid ``observed_relative_error`` (finite, >= 0, linear flux only)."""
    if observed_relative_error is None:
        return
    if not (np.isfinite(observed_relative_error) and observed_relative_error >= 0.0):
        raise ValueError(
            f"observed_relative_error must be finite and >= 0 (CIGALE: 'The relative "
            f"model error must be positive'); got {observed_relative_error!r}."
        )
    if flux_unit == "ab_mag":
        raise ValueError(
            "observed_relative_error scales the linear observed flux; it is not "
            "defined for flux_unit='ab_mag'. Convert the table to a flux unit first."
        )


def _apply_observed_floor(flux, err, presence, censor, fraction):
    r"""Add an observed-flux-scaled relative error in quadrature on the detections.

    .. math::

        \sigma_{\rm eff}^2 = \sigma^2 + (f_{\rm obs}\, \epsilon)^2

    for each band with ``presence`` true, a non-negative error and a censor flag of
    zero (CIGALE ``_add_model_error``, Boquien et al. 2019 Sect. 4.3). Limits and absent
    bands keep ``err``. Returns a new array; the inputs are not modified.

    Parameters
    ----------
    flux : ndarray, shape (N, n_bands)
        Observed flux [erg/s/cm^2/Hz], already converted.
    err : ndarray, shape (N, n_bands)
        1-sigma error, same unit as ``flux``.
    presence : ndarray of bool, shape (N, n_bands)
        True where the band is present.
    censor : ndarray of int, shape (N, n_bands), or None
        Censor flags (0 detected, 1 upper limit, -1 lower limit); None if absent.
    fraction : float or None
        Relative error :math:`\epsilon`; None returns ``err`` unchanged.

    Returns
    -------
    ndarray, shape (N, n_bands)
        Effective 1-sigma error, same unit as ``err``.

    Notes
    -----
    Not JIT-traced: runs once on host arrays during ingest.
    """
    if fraction is None:
        return err
    detected = presence & (err >= 0.0)
    if censor is not None:
        detected = detected & (censor == 0)
    inflated = np.sqrt(err**2 + (fraction * np.abs(flux)) ** 2)
    return np.where(detected, inflated, err)


def _zero_non_finite(values: np.ndarray) -> np.ndarray:
    """Replace every NaN and +/-inf by 0.0; finite values are returned unchanged."""
    return np.where(np.isfinite(values), values, 0.0)


def _validate_cigale_input_options(default_relative_error, lim_flag, flux_unit, censor_cols):
    """Refuse an inconsistent ``default_relative_error`` / ``lim_flag`` request."""
    if default_relative_error is not None and not default_relative_error >= 0.0:
        raise ValueError(
            f"default_relative_error must be >= 0 (CIGALE: 'The relative default "
            f"error must be positive'); got {default_relative_error!r}."
        )
    if lim_flag is not None and lim_flag not in LIM_FLAGS:
        raise ValueError(f"lim_flag={lim_flag!r} not in {list(LIM_FLAGS)}.")
    if (default_relative_error is not None or lim_flag is not None) and flux_unit == "ab_mag":
        raise ValueError(
            "default_relative_error and lim_flag encode CIGALE's linear-flux "
            "tables; they are not defined for flux_unit='ab_mag'. Convert the "
            "table to a flux unit first."
        )
    if lim_flag in ("noscaling", "full") and censor_cols is not None:
        raise ValueError(
            f"lim_flag={lim_flag!r} derives the censor flags from negative "
            "errors; it cannot be combined with censor_cols (two sources of "
            "censor flags). Pass one of them."
        )
