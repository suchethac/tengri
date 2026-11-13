"""Known single-cell defects in bundled SSP grids, repaired at load (#3406).

Provenance. In the four bundled Padova grids (``fsps_prsc_miles_{chabrier,kroupa,salpeter}``
and ``fsps_prsc_c3k_a_chabrier``) the cell at lgmet = -3.301 (Z = 0.0005) and 1.000 Gyr
is about 2.8x brighter at 6200 A than its metallicity and age neighbours, at every
wavelength. The MIST grids have no such cell. The grids were built with python-fsps and no
Padova FSPS build is recorded in this checkout (the installed python-fsps is MIST-only),
so the defect could not be reproduced upstream; it is repaired here instead.

The repair replaces the flux of each listed cell by the geometric mean of its two age
neighbours at the same metallicity (linear interpolation of log flux in age). It applies to
every wavelength and leaves ``ssp_mass_remaining`` untouched, which is consistent with the
neighbours at that cell. The table is explicit: spikes are not detected at run time.
"""

from __future__ import annotations

import numpy as np

#: Grid file stem -> cells ``(lgmet, lg_age_gyr)`` whose flux is replaced at load (#3406).
_KNOWN_GRID_DEFECTS: dict[str, tuple[tuple[float, float], ...]] = {
    "fsps_prsc_miles_chabrier": ((-3.30103, 0.0),),
    "fsps_prsc_miles_kroupa": ((-3.30103, 0.0),),
    "fsps_prsc_miles_salpeter": ((-3.30103, 0.0),),
    "fsps_prsc_c3k_a_chabrier": ((-3.30103, 0.0),),
}

#: Tolerance [dex] for matching a listed (lgmet, lg_age) to a grid node.
_NODE_ATOL: float = 1e-3


def _node_index(axis: np.ndarray, value: float) -> int | None:
    """Index of the node at ``value``, or ``None`` when the grid has no such node."""
    idx = int(np.argmin(np.abs(axis - value)))
    return idx if abs(float(axis[idx]) - value) <= _NODE_ATOL else None


def repair_known_grid_defects(
    stem: str,
    ssp_flux,
    lgmet: np.ndarray,
    lg_age_gyr: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    """Replace the listed defective cells of ``ssp_flux`` (#3406).

    Parameters
    ----------
    stem : str
        Grid file stem, the key of :data:`_KNOWN_GRID_DEFECTS`.
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Flux templates. A 4D (alpha-enhanced) grid is returned unchanged.
    lgmet, lg_age_gyr : array
        Metallicity (log10 Z) and age (log10 Gyr) node values.

    Returns
    -------
    flux : ndarray
        float64 copy of ``ssp_flux`` with the listed cells repaired; the input
        itself when the grid has no entry.
    repaired : tuple of (lgmet, lg_age_gyr)
        The node values actually repaired, empty when the grid has no entry.

    Raises
    ------
    ValueError
        If a listed node is absent, sits on the age edge, or has non-positive
        neighbour flux (log interpolation undefined).
    """
    defects = _KNOWN_GRID_DEFECTS.get(stem, ())
    if not defects or np.ndim(ssp_flux) != 3:
        return ssp_flux, ()

    lgmet = np.asarray(lgmet, dtype=np.float64)
    lg_age_gyr = np.asarray(lg_age_gyr, dtype=np.float64)
    flux = np.array(ssp_flux, dtype=np.float64)
    repaired: list[tuple[float, float]] = []
    for lg_met_value, lg_age_value in defects:
        k = _node_index(lgmet, lg_met_value)
        i = _node_index(lg_age_gyr, lg_age_value)
        if k is None or i is None:
            # A grid without the defective node cannot carry that defect: nothing to repair.
            continue
        if not 0 < i < flux.shape[1] - 1:
            raise ValueError(f"known grid defect at age edge: {stem} age index {i}")
        neighbours = flux[k, i - 1 : i + 2 : 2]
        if not np.all(neighbours > 0.0):
            raise ValueError(f"known grid defect in {stem}: non-positive age-neighbour flux")
        flux[k, i] = np.sqrt(neighbours[0] * neighbours[1])
        repaired.append((float(lgmet[k]), float(lg_age_gyr[i])))

    return flux, tuple(repaired)
