# SPDX-License-Identifier: BSD-3-Clause
"""The 4000 Angstrom break windows: published in air, applied in vacuum.

Bruzual (1983, ApJ 273, 105) defined ``D4000`` and Balogh et al. (1999, ApJ 527,
54, Table 1) defined the narrow ``Dn4000`` on ground-based spectra, whose
published wavelengths are air wavelengths: Balogh et al. center their
[O II] window on the air name ``lambda3727`` (the vacuum doublet centroid is
3728.5), and the SDSS-MaNGA pipeline paper (Westfall et al. 2019, AJ 158, 231,
Table 4, rows 44-45) lists both break passbands as defined for air. Like the Lick
windows, they are stored here exactly as published (air) and converted once, at
import, with :func:`tengri.utils.air_vacuum.air_to_vac`, so every consumer of a
break (the index catalog, ``dn4000``, ``compute_dn4000``) sees the same vacuum
windows. This module is the single source of those numbers.
"""

from __future__ import annotations

import numpy as np

from tengri.utils.air_vacuum import air_to_vac

__all__ = [
    "BREAK_AIR_WINDOWS",
    "BREAK_VACUUM_WINDOWS",
    "DN4000_BLUE_AA",
    "DN4000_RED_AA",
]

#: Published air windows ``name -> (blue continuum, red continuum)`` [Angstrom].
BREAK_AIR_WINDOWS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "Dn4000": ((3850.0, 3950.0), (4000.0, 4100.0)),
    "D4000": ((3750.0, 3950.0), (4050.0, 4250.0)),
}


def _to_vacuum(window: tuple[float, float]) -> tuple[float, float]:
    lo, hi = (float(w) for w in air_to_vac(np.asarray(window, dtype=np.float64)))
    return (lo, hi)


#: The same windows in vacuum Angstrom (what tengri's spectra are in).
BREAK_VACUUM_WINDOWS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    name: (_to_vacuum(blue), _to_vacuum(red)) for name, (blue, red) in BREAK_AIR_WINDOWS.items()
}

DN4000_BLUE_AA: tuple[float, float] = BREAK_VACUUM_WINDOWS["Dn4000"][0]
DN4000_RED_AA: tuple[float, float] = BREAK_VACUUM_WINDOWS["Dn4000"][1]
