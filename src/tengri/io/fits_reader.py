# SPDX-License-Identifier: BSD-3-Clause
"""Generic FITS spectrum reader with column name auto-detection."""

from __future__ import annotations

import numpy as np

from tengri.io.arrays import SpectrumTuple


def read_generic_fits_spectrum(
    path: str,
    wave_col: str = "WAVELENGTH",
    flux_col: str = "FLUX",
    err_col: str | None = "ERROR",
    hdu: int | str = 1,
) -> SpectrumTuple:
    """Read a generic FITS spectrum from a table-HDU.

    Detects common column name variants and auto-converts inverse variance
    to 1-sigma error. Wavelengths are assumed to be in Angstrom; flux is
    assumed to be in erg/s/cm^2/A.

    Parameters
    ----------
    path: str
        Path to FITS file.
    wave_col: str, optional
        Primary wavelength column name. Auto-detected if not found;
        tries WAVE, wavelength, LAMBDA, lambda. Default: 'WAVELENGTH'.
    flux_col: str, optional
        Primary flux column name. Auto-detected if not found;
        tries FLUX_DENSITY, FLUX_OBS, flux. Default: 'FLUX'.
    err_col: str or None, optional
        Error column name. If 'IVAR', converts inverse variance to
        1-sigma error. If None, creates NaN-filled error array.
        Auto-detected if not found; tries ERROR, FLUX_ERR, ERR.
        Default: 'ERROR'.
    hdu: int or str, optional
        HDU index or name. Default: 1 (first extension).

    Returns
    -------
    SpectrumTuple
        Tuple of (wave, flux, flux_err, meta) where meta contains
        'instrument' and 'header_keys'.

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If required columns cannot be found or data is malformed.

    Notes
    -----
    **JIT-compatible**: no, file I/O and astropy required.

    Examples
    --------
    >>> spec = read_generic_fits_spectrum("spectrum.fits")
    >>> wave, flux, err, meta = spec
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError(
            "read_generic_fits_spectrum requires astropy: pip install astropy"
        ) from None

    with fits.open(path) as hdul:
        hdu_data = hdul[hdu].data
        hdu_header = hdul[hdu].header

    if hdu_data is None:
        raise ValueError(f"HDU {hdu} is empty or not a binary table")

    colnames_lower = {name.upper(): name for name in hdu_data.dtype.names}

    def find_col(primary: str, variants: list[str]) -> str | None:
        """Find column by primary name or variants (case-insensitive)."""
        for name in [primary.upper()] + [v.upper() for v in variants]:
            if name in colnames_lower:
                return colnames_lower[name]
        return None

    wave_col_name = find_col(wave_col, ["WAVE", "wavelength", "LAMBDA", "lambda"])
    if not wave_col_name:
        raise ValueError(
            f"Could not find wavelength column. Available: {list(hdu_data.dtype.names)}"
        )

    flux_col_name = find_col(flux_col, ["FLUX_DENSITY", "FLUX_OBS", "flux"])
    if not flux_col_name:
        raise ValueError(f"Could not find flux column. Available: {list(hdu_data.dtype.names)}")

    wave = np.asarray(hdu_data[wave_col_name], dtype=np.float64)
    flux = np.asarray(hdu_data[flux_col_name], dtype=np.float64)

    if err_col is None:
        flux_err = np.full_like(flux, np.nan)
    elif err_col.upper() == "IVAR":
        ivar = np.asarray(hdu_data["IVAR"], dtype=np.float64)
        flux_err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
    else:
        err_col_name = find_col(err_col, ["FLUX_ERR", "FLUX_ERROR", "ERR", "ERROR"])
        if err_col_name:
            flux_err = np.asarray(hdu_data[err_col_name], dtype=np.float64)
        else:
            flux_err = np.full_like(flux, np.nan)

    meta = {
        "instrument": hdu_header.get("INSTRUME", "unknown"),
        "header_keys": dict(hdu_header),
    }

    return SpectrumTuple(wave, flux, flux_err, meta)
