# SPDX-License-Identifier: BSD-3-Clause
"""Shared caller grids and the ln(nu) reference integral for the AGN publication tests."""

from __future__ import annotations

import numpy as np

_C_AA = 2.99792458e18
PUB_DENSE = np.geomspace(10.0, 1.0e7, 4000)
PUB_COARSE = np.geomspace(10.0, 1.0e7, 40)
PUB_FILTERS = np.array(
    [
        1528.0,
        2271.0,
        3551.0,
        4686.0,
        6166.0,
        7480.0,
        8932.0,
        12350.0,
        16620.0,
        21590.0,
        33526.0,
        46028.0,
    ]
)
PUB_GRIDS = (PUB_DENSE, PUB_COARSE, PUB_FILTERS)


def ln_nu_integral(sed, wave):
    """Bolometric integral of L_nu on any grid: trapezoid in ln(nu) of L_nu * nu [erg/s]."""
    nu = _C_AA / np.asarray(wave, dtype=float)
    order = np.argsort(nu)
    y = np.asarray(sed, dtype=float)[order] * nu[order]
    return float(np.trapezoid(y, np.log(nu[order])))
