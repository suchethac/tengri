"""DESI spectrum reader."""

from __future__ import annotations

import numpy as np

from tengri.io.arrays import SpectrumTuple


def read_desi(path: str) -> SpectrumTuple:
    """Read a DESI coadd or healpix spectrum into a single combined spectrum.

    Looks for BRZ-combined wavelength/flux/ivar HDUs or falls back to
    per-arm (b/r/z) HDUs. Concatenates and sorts by wavelength.

    Parameters
    ----------
    path : str
        Path to DESI FITS file (e.g., coadd-*.fits or similar).

    Returns
    -------
    SpectrumTuple
        Tuple of (wave, flux, flux_err, meta) where wave/flux/flux_err
        are sorted by wavelength. meta includes 'instrument' ('DESI'),
        'redshift' (if present), and 'header_keys'.

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If neither combined nor per-arm HDUs are found.

    Notes
    -----
    **JIT-compatible**: no — file I/O and astropy required.

    DESI spectra are often provided in combined BRZ HDUs (e.g.,
    BRZ_WAVELENGTH, BRZ_FLUX, BRZ_IVAR) or as separate b/r/z arms.
    This reader concatenates multiple arms and sorts.

    Examples
    --------
    >>> spec = read_desi("coadd-123-45.fits")
    >>> wave, flux, err, meta = spec
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("read_desi requires astropy: pip install astropy") from None

    with fits.open(path) as hdul:
        hdr_names = [hdu.name.upper() if hdu.name else f"HDU{i}" for i, hdu in enumerate(hdul)]
        primary_header = hdul[0].header

        wave_list = []
        flux_list = []
        ivar_list = []

        if "BRZ_WAVELENGTH" in hdr_names:
            idx = hdr_names.index("BRZ_WAVELENGTH")
            hdu_data = hdul[idx].data
            if hdu_data.dtype.names:
                wave_list.append(np.asarray(hdu_data["BRZ_WAVELENGTH"], dtype=np.float64))
                flux_list.append(np.asarray(hdu_data["BRZ_FLUX"], dtype=np.float64))
                if "BRZ_IVAR" in hdu_data.dtype.names:
                    ivar_list.append(np.asarray(hdu_data["BRZ_IVAR"], dtype=np.float64))
            else:
                flux_idx = hdr_names.index("BRZ_FLUX")
                wave_list.append(np.asarray(hdul[idx].data, dtype=np.float64))
                flux_list.append(np.asarray(hdul[flux_idx].data, dtype=np.float64))
                if "BRZ_IVAR" in hdr_names:
                    ivar_idx = hdr_names.index("BRZ_IVAR")
                    ivar_list.append(np.asarray(hdul[ivar_idx].data, dtype=np.float64))
        else:
            for arm in ["B", "R", "Z"]:
                wave_hdu = f"{arm}_WAVELENGTH"
                flux_hdu = f"{arm}_FLUX"
                ivar_hdu = f"{arm}_IVAR"

                if wave_hdu in hdr_names:
                    idx = hdr_names.index(wave_hdu)
                    hdu_data = hdul[idx].data
                    if hdu_data.dtype.names:
                        wave_list.append(np.asarray(hdu_data["WAVELENGTH"], dtype=np.float64))
                        flux_list.append(np.asarray(hdu_data["FLUX"], dtype=np.float64))
                        if "IVAR" in hdu_data.dtype.names:
                            ivar_list.append(np.asarray(hdu_data["IVAR"], dtype=np.float64))
                    else:
                        flux_idx = hdr_names.index(flux_hdu)
                        ivar_idx = hdr_names.index(ivar_hdu)
                        wave_list.append(np.asarray(hdul[idx].data, dtype=np.float64))
                        flux_list.append(np.asarray(hdul[flux_idx].data, dtype=np.float64))
                        if ivar_idx >= 0:
                            ivar_list.append(np.asarray(hdul[ivar_idx].data, dtype=np.float64))

        if not wave_list:
            raise ValueError(
                f"Could not find BRZ or per-arm wavelength HDUs. Available: {hdr_names}"
            )

    wave = np.concatenate(wave_list)
    flux = np.concatenate(flux_list)

    if ivar_list:
        ivar = np.concatenate(ivar_list)
        flux_err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
    else:
        flux_err = np.full_like(flux, np.nan)

    sort_idx = np.argsort(wave)
    wave = wave[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]

    meta = {
        "instrument": "DESI",
        "redshift": primary_header.get("Z", None),
        "header_keys": dict(primary_header),
    }

    return SpectrumTuple(wave, flux, flux_err, meta)
