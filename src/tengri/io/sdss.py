# SPDX-License-Identifier: BSD-3-Clause
"""SDSS/BOSS/eBOSS spectrum reader."""

from __future__ import annotations

import numpy as np

from tengri.io.arrays import SpectrumTuple


def read_sdss(path: str) -> SpectrumTuple:
    """Read an SDSS/BOSS/eBOSS spectrum from a spec-lite FITS file.

    Converts log10(lambda) in vacuum to linear wavelength in Angstrom.
    Converts inverse variance (IVAR) to 1-sigma error. Handles the
    standard FITS binary table layout used in SDSS DR7+ spec-lite files
    and spPlate spectra.

    Parameters
    ----------
    path : str
        Path to FITS file (e.g., spec-*.fits or spPlate-*.fits).

    Returns
    -------
    SpectrumTuple
        Tuple of (wave, flux, flux_err, meta) where meta includes
        'redshift' (if present in header), 'instrument' ('SDSS' or variant),
        and 'header_keys'.

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If expected SDSS columns are not found.

    Notes
    -----
    **JIT-compatible**: no, file I/O and astropy required.

    The SDSS spec-lite format uses LOGLAM (log10 of vacuum wavelength)
    and FLUX (counts or normalized). IVAR is inverse variance (1/sigma^2).

    Examples
    --------
    >>> spec = read_sdss("spec-1234-56789-0001.fits")
    >>> wave, flux, err, meta = spec
    >>> z = meta.get("redshift")
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("read_sdss requires astropy: pip install astropy") from None

    with fits.open(path) as hdul:
        data = hdul[1].data
        header = hdul[1].header

    if data is None:
        raise ValueError("No data in primary extension")

    colnames = {name.upper(): name for name in data.dtype.names}

    if "LOGLAM" not in colnames:
        raise ValueError(
            f"LOGLAM column not found in SDSS file. Available columns: {list(data.dtype.names)}"
        )

    if "FLUX" not in colnames:
        raise ValueError(
            f"FLUX column not found in SDSS file. Available columns: {list(data.dtype.names)}"
        )

    loglam = np.asarray(data[colnames["LOGLAM"]], dtype=np.float64)
    wave = 10.0**loglam

    flux = np.asarray(data[colnames["FLUX"]], dtype=np.float64)

    if "IVAR" in colnames:
        ivar = np.asarray(data[colnames["IVAR"]], dtype=np.float64)
        flux_err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
    else:
        flux_err = np.full_like(flux, np.nan)

    meta = {
        "instrument": "SDSS",
        "redshift": header.get("Z", None),
        "header_keys": dict(header),
    }

    return SpectrumTuple(wave, flux, flux_err, meta)
