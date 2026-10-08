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

__all__ = ["CLOUDY_LABEL_AIR_MAX_AA", "LABEL_AIR_MAX_AA", "catalog_air_to_vacuum"]

#: Longest wavelength [Angstrom] that PyNeb/MAPPINGS-style integer labels quote
#: in air. The Flury+2024 infrared labels (12.81, 25.89, 88.36, 157.74 micron,
#: ...) are the NIST *vacuum* values: [O III] 88.36 and [C II] 157.74 are the
#: vacuum wavelengths, the air ones being 88.33 and 157.70. Short labels
#: (< 2000 Angstrom) follow the IAU cutoff inside
#: :func:`~tengri.utils.air_vacuum.air_to_vac`.
LABEL_AIR_MAX_AA: float = 1.0e4

#: Longest wavelength [Angstrom] that *Cloudy* line labels quote in air. Cloudy
#: converts every label longer than 2000 Angstrom to air, including the
#: near- and mid-infrared ones: the shipped Cue and Synthesizer catalogs carry
#: Pa-alpha 18751.0, Pa-beta 12818.1, Br-gamma 21655.3 (vacuum 18756.1,
#: 12821.7, 21661.2) and [Ne II] 12.8101, [Ne III] 15.5509, [S IV] 10.5076,
#: [O III] 88.3323 micron (vacuum 12.8136, 15.5551, 10.5105, 88.3564). After
#: ``air_to_vac`` all the labels checked from 1.2 to 100 micron agree with the
#: vacuum value to 5e-5 or better (unconverted: 2.7e-4; checked in
#: ``test_bug_vacuum_conventions_remaining``). The 100 micron cap stops short of
#: the far-infrared labels ([O I] 145.495, [N II] 121.769 and 205.283, [C II]
#: 157.636, [C I] 370/609 micron): some match the vacuum value after
#: ``air_to_vac`` and others differ from it by up to 1e-3 in either direction
#: (atomic data, not refraction), so one rule cannot be established for them.
CLOUDY_LABEL_AIR_MAX_AA: float = 1.0e6


def catalog_air_to_vacuum(
    waves, *, air_mask=None, max_air_aa: float = LABEL_AIR_MAX_AA
) -> np.ndarray:
    """Convert the air entries of a line catalog to vacuum [Angstrom].

    Parameters
    ----------
    waves : array_like, shape (n_lines,)
        Rest-frame line wavelengths as read from the catalog [Angstrom].
    air_mask : array_like of bool, shape (n_lines,), optional
        Which entries the catalog quotes in air. ``None`` means every entry
        inside the label window (2000 Angstrom to ``max_air_aa``);
        pass a mask for a catalog that mixes air and vacuum lines (the
        Feltre+2016 table).
    max_air_aa : float, optional
        Upper end of the label window [Angstrom]. Defaults to
        :data:`LABEL_AIR_MAX_AA`; Cloudy-lineage catalogs pass
        :data:`CLOUDY_LABEL_AIR_MAX_AA`.

    Returns
    -------
    ndarray, shape (n_lines,)
        ``float64`` vacuum wavelengths [Angstrom]; entries outside the label
        window, or not flagged by ``air_mask``, are returned unchanged.
    """
    wave = np.asarray(waves, dtype=np.float64)
    in_window = (wave >= 2000.0) & (wave <= max_air_aa)
    if air_mask is not None:
        in_window = in_window & np.asarray(air_mask, dtype=bool)
    return np.where(in_window, air_to_vac(wave), wave)
