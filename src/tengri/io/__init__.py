# SPDX-License-Identifier: BSD-3-Clause
"""I/O primitives for loading spectra and photometry from common formats.

All readers return ``(wave_angstrom, flux, flux_err, meta)`` tuples with
standard units (erg/s/cm^2/A for flux). ``meta`` carries redshift (when known),
instrument name, original FITS header keywords of interest.

Optional dependencies (astropy, specutils, requests) are imported lazily — a
missing dependency produces a clear ImportError with install guidance.
"""

from __future__ import annotations

from tengri.io.arrays import SpectrumTuple
from tengri.io.catalog import load_catalog
from tengri.io.desi import (
    DesiCamera,
    desi_resolution_matrix,
    desi_spectroscopy,
    read_desi,
    read_desi_cameras,
)
from tengri.io.fits_reader import read_generic_fits_spectrum
from tengri.io.sdss import read_sdss
from tengri.io.specutils_bridge import from_spectrum1d

__all__ = [
    "DesiCamera",
    "SpectrumTuple",
    "desi_resolution_matrix",
    "desi_spectroscopy",
    "from_spectrum1d",
    "load_catalog",
    "read_desi",
    "read_desi_cameras",
    "read_generic_fits_spectrum",
    "read_sdss",
]
