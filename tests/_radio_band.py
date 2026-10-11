# SPDX-License-Identifier: BSD-3-Clause
"""Independent band luminosity of the dust-emission SED, for forward-derived radio pins.

The radio block integrates ``sed_dust_ir`` with the edge-exact cell integral of
``tengri.utils.sed_quantities.log10_band_luminosity``. A test that pins the radio flux
must not reuse it, so this forms the same quantity by a different route: a dense
log-frequency resampling of the SED, then a trapezoid.
"""

from __future__ import annotations

import numpy as np

from tengri.components.radio.component import IR_WINDOWS_AA
from tengri.utils.physics_constants import C_AA


def band_l_ir(state, window: str = "tir") -> float:
    """L(window) [erg/s] of ``state.derived['sed_dust_ir']`` by dense interpolation."""
    lo, hi = IR_WINDOWS_AA[window]
    wave = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_dust_ir"])
    sel = (wave > 1.0e3) & (wave < 5.0e8)
    nu = np.geomspace(C_AA / hi, C_AA / lo, 400001)
    f = np.interp(nu, (C_AA / wave[sel])[::-1], sed[sel][::-1])
    return float(np.trapezoid(f, nu))
