# SPDX-License-Identifier: BSD-3-Clause
"""Ingestion-time air -> vacuum conversion of tabulated emission-line catalogs.

Every wavelength inside tengri is a vacuum wavelength. Photoionization grids
inherit line *labels* from Cloudy, PyNeb or the NEOGAL tables, which quote air
wavelengths above 2000 Angstrom (Cloudy) and vacuum below. A loader that reads
such a catalog calls :func:`catalog_air_to_vacuum` exactly once, on the array
it read from disk, so everything downstream (line placement in a filter, line
matching against a vacuum target, flux measurement) sees vacuum. The
conversion is :func:`tengri.utils.air_vacuum.air_to_vac`; this module never
carries a refractive-index formula of its own.
"""

from __future__ import annotations

import numpy as np

from tengri.utils.air_vacuum import air_to_vac

__all__ = ["LABEL_AIR_MAX_AA", "catalog_air_to_vacuum"]

#: Longest wavelength [Angstrom] that photoionization-code line labels quote in
#: air. Longer labels (the infrared fine-structure lines, quoted in microns)
#: are vacuum by construction, and short ones (< 2000 Angstrom) follow the IAU
#: cutoff inside :func:`~tengri.utils.air_vacuum.air_to_vac`.
LABEL_AIR_MAX_AA: float = 1.0e4


def catalog_air_to_vacuum(waves, *, air_mask=None) -> np.ndarray:
    """Convert the air entries of a line catalog to vacuum [Angstrom].

    Parameters
    ----------
    waves : array_like, shape (n_lines,)
        Rest-frame line wavelengths as read from the catalog [Angstrom].
    air_mask : array_like of bool, shape (n_lines,), optional
        Which entries the catalog quotes in air. ``None`` means every entry
        inside the label window (2000 Angstrom to :data:`LABEL_AIR_MAX_AA`);
        pass a mask for a catalog that mixes air and vacuum lines (the
        Feltre+2016 table).

    Returns
    -------
    ndarray, shape (n_lines,)
        ``float64`` vacuum wavelengths [Angstrom]; entries outside the label
        window, or not flagged by ``air_mask``, are returned unchanged.
    """
    wave = np.asarray(waves, dtype=np.float64)
    in_window = (wave >= 2000.0) & (wave <= LABEL_AIR_MAX_AA)
    if air_mask is not None:
        in_window = in_window & np.asarray(air_mask, dtype=bool)
    return np.where(in_window, air_to_vac(wave), wave)
