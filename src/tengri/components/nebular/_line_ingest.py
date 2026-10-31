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

from functools import cache

import numpy as np

from tengri.observation.line_list import LineList
from tengri.utils.air_vacuum import air_to_vac, vac_to_air

__all__ = [
    "CLOUDY_BLEND_LABEL_LINES",
    "CLOUDY_LABEL_AIR_MAX_AA",
    "LABEL_AIR_MAX_AA",
    "catalog_air_to_vacuum",
    "cloudy_labels_to_vacuum",
    "rounded_labels_to_vacuum",
]

#: Longest wavelength [Angstrom] that PyNeb/MAPPINGS-style integer labels quote
#: in air. The Flury+2024 infrared labels (12.81, 25.89, 88.36, 157.74 micron,
#: ...) are the NIST *vacuum* values: [O III] 88.36 and [C II] 157.74 are the
#: vacuum wavelengths, the air ones being 88.33 and 157.70. Short labels
#: (< 2000 Angstrom) follow the IAU cutoff inside
#: :func:`~tengri.utils.air_vacuum.air_to_vac`.
LABEL_AIR_MAX_AA: float = 1.0e4

#: Longest wavelength [Angstrom] that *Cloudy* line labels quote in air: none.
#: Cloudy stores every wavelength in vacuum and *prints* it in air unless
#: ``print line vacuum`` is given. The rule is in the source (cloudy-master
#: ``source/cddefines.cpp``, ``t_wavl::sprt_wl``: ``if
#: (prt.lgPrintLineAirWavelengths && wl > 0) wl /= p_RefIndex(1e8/wl)``;
#: ``prt.lgPrintLineAirWavelengths`` defaults to true in
#: ``init_defaults_preparse.cpp``; ``p_RefIndex`` applies the Peck & Reeder
#: 1972 index wherever the wavenumber is below 5e4 cm^-1, i.e. longward of
#: 2000 Angstrom, with no upper limit; ``t_wavl::p_wlAirVac`` has the same
#: 2000 Angstrom floor). Micron and centimetre labels are the same air value
#: in other units. The shipped Cue and Synthesizer catalogs carry Pa-alpha
#: 18751.0, Pa-beta 12818.1, Br-gamma 21655.3 (vacuum 18756.1, 12821.7,
#: 21661.2) and [Ne II] 12.8101, [Ne III] 15.5509, [S IV] 10.5076, [O III]
#: 88.3323 micron (vacuum 12.8136, 15.5551, 10.5105, 88.3564); after
#: ``air_to_vac`` those agree with the vacuum value to 5e-5 or better
#: (unconverted: 2.7e-4). Beyond 100 micron the index tends to the constant
#: 1 + 2.726e-4, so the far-infrared labels ([N II] 121.769, [O I] 145.495,
#: [C II] 157.636, [N II] 205.283, [C I] 370 and 609 micron) are air labels
#: too. Some then differ from the NIST vacuum value by up to 1e-3; that is
#: Cloudy's own level energies, not a medium. Cloudy's index (Peck & Reeder)
#: and :func:`~tengri.utils.air_vacuum.air_to_vac` (Morton 2000) differ by
#: less than 1e-7 there. The 2000 Angstrom floor is applied inside
#: :func:`catalog_air_to_vacuum`.
CLOUDY_LABEL_AIR_MAX_AA: float = float("inf")


#: Cloudy's labels for the [O II] 7320 and 7330 auroral blends, and the
#: ``observation/line_list.py`` line whose vacuum blend center each stands for.
#: These are blend names, not measured wavelengths: the air blends sit near
#: 7320.5 and 7330.2 Angstrom, 2.5 and 1.8 Angstrom from the labels, so
#: converting the label as an air wavelength lands 3.55 and 1.81 Angstrom red of
#: the vacuum centers (#3113). The centers come from the line list, not from here.
CLOUDY_BLEND_LABEL_LINES: dict[float, str] = {
    7323.0: "OII_7320",
    7332.0: "OII_7330",
    7332.15: "OII_7330",
}

_BLEND_LABEL_TOL_AA = 1.0e-3


@cache
def _catalog_vacuum_aa() -> tuple[tuple[str, ...], np.ndarray]:
    """Names and vacuum wavelengths [Angstrom] of the line-list catalog."""
    lines = LineList.default_optical()
    return lines.names, np.asarray(lines.wavelengths, dtype=np.float64)


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


def cloudy_labels_to_vacuum(waves, *, max_air_aa: float = CLOUDY_LABEL_AIR_MAX_AA) -> np.ndarray:
    """Convert Cloudy line labels to vacuum [Angstrom], blend labels to their centers.

    A Cloudy label is an air wavelength (:func:`catalog_air_to_vacuum`) except
    the blend names in :data:`CLOUDY_BLEND_LABEL_LINES`, which are placed at the
    vacuum blend center of the line list instead of being converted.
    """
    raw = np.asarray(waves, dtype=np.float64)
    vac = catalog_air_to_vacuum(raw, max_air_aa=max_air_aa)
    names, centers = _catalog_vacuum_aa()
    for label, name in CLOUDY_BLEND_LABEL_LINES.items():
        vac = np.where(np.abs(raw - label) < _BLEND_LABEL_TOL_AA, centers[names.index(name)], vac)
    return vac


def rounded_labels_to_vacuum(
    waves,
    *,
    half_width_aa: float,
    blends: bool = False,
    air_mask=None,
    max_air_aa: float = LABEL_AIR_MAX_AA,
) -> np.ndarray:
    """Convert integer-rounded air labels to the vacuum line they round.

    PyNeb/MAPPINGS-style labels (``H1r_6563A``, ``O3_5007A``) round the air
    wavelength to the nearest Angstrom, so converting the label leaves a residual
    of up to ``half_width_aa`` on top of the exact vacuum line. Each air-window
    label within ``half_width_aa`` of the air wavelength of a line-list line takes
    that line's vacuum wavelength; the nearest line wins. Labels with no such
    line take the plain conversion. The two [O II] blend lines are candidates
    only when ``blends`` is true: a catalog that lists the doublet components
    separately (Flury: 7319, 7320, 7330, 7331) names single lines, not centers.

    Parameters
    ----------
    waves : array_like, shape (n_lines,)
        Rounded air labels [Angstrom].
    half_width_aa : float
        Largest distance [Angstrom] between a label and the air wavelength of
        the line it rounds.
    blends : bool, optional
        Whether the [O II] 7320/7330 blend centers are candidates (a catalog whose
        7320 and 7330 labels name the whole blend).
    air_mask, max_air_aa
        As in :func:`catalog_air_to_vacuum`.
    """
    raw = np.asarray(waves, dtype=np.float64)
    vac = catalog_air_to_vacuum(raw, air_mask=air_mask, max_air_aa=max_air_aa)
    names, centers = _catalog_vacuum_aa()
    single = np.array([blends or n not in CLOUDY_BLEND_LABEL_LINES.values() for n in names])
    centers = centers[single]
    dist = np.abs(vac_to_air(centers)[None, :] - raw[:, None])
    nearest = np.argmin(dist, axis=1)
    in_window = vac != raw
    hit = in_window & (dist[np.arange(raw.size), nearest] <= half_width_aa)
    return np.where(hit, centers[nearest], vac)
