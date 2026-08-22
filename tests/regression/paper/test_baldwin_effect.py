# SPDX-License-Identifier: BSD-3-Clause
"""Regression: tengri's qsogen disc reproduces the Baldwin effect.

The Baldwin effect (Baldwin 1977 [1]_) is the inverse correlation between a
quasar's broad emission-line equivalent width and its continuum luminosity:
more luminous quasars have *relatively weaker* lines. qsogen (Temple, Hewett &
Banerji 2021 [2]_ -- the model AGNfitter-RX tabulates as "THB21") encodes it by
scaling line strength with absolute magnitude / luminosity.

This pins that tengri's ``qsogen`` disc + line blocks reproduce the trend: the
Halpha+[N II] 0.7 um bump contrast L_nu(6563)/L_nu(2500) must *decrease*
monotonically with ``agn_log_lbol``. The contrast is an AGN-only quantity
(independent of the stellar SSP), so this runs on the synthetic CI SSP.

References
----------
.. [1] J. A. Baldwin, "Luminosity Indicators in the Spectra of Quasi-Stellar
   Objects," ApJ, 214, 679 (1977). https://doi.org/10.1086/155294
.. [2] M. J. Temple, P. C. Hewett & M. Banerji, "Exploring quasar SEDs:
   new constraints on the BBB," MNRAS, 508, 737 (2021).
   https://doi.org/10.1093/mnras/stab2586
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

tengri = pytest.importorskip("tengri")


def _halpha_contrast(model) -> float:
    """L_nu(6563 A) / L_nu(2500 A) of the AGN SED (the 0.7 um line bump)."""
    state = model.predict_state({})
    w = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_agn"])
    order = np.argsort(w)
    ref = np.interp(2500.0, w[order], sed[order])
    return float(np.interp(6563.0, w[order], sed[order]) / ref)


def _build(ssp, log_lbol):
    from tengri import FIXED, Fixed, SEDModel

    return SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0), "all_params": FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": FIXED,
        },
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": FIXED},
            "torus": {"type": "none"},
            "lines": {"type": "qsogen", "all_params": FIXED},
            "feii": {"type": "qsogen_balmer", "all_params": FIXED},
            "agn_log_lbol": Fixed(log_lbol),
            "all_params": FIXED,
        },
        redshift=Fixed(0.0),
    )


def test_qsogen_reproduces_baldwin_anticorrelation():
    """Hα/2500 Å line contrast decreases monotonically with luminosity."""
    ssp = tengri.load_ssp()
    log_lbols = [11.0, 12.0, 13.0, 14.0, 15.0]
    contrasts = [_halpha_contrast(_build(ssp, lb)) for lb in log_lbols]

    # Monotonically decreasing = the Baldwin anti-correlation.
    diffs = np.diff(contrasts)
    assert np.all(diffs < 0.0), f"line contrast not monotically decreasing with L: {contrasts}"

    # The faint end has materially stronger lines than the bright end (a real
    # Baldwin slope, not numerical drift): >25% drop across 4 dex in L_bol.
    assert contrasts[0] / contrasts[-1] > 1.25, (
        f"Baldwin slope too shallow: faint/bright contrast ratio "
        f"{contrasts[0] / contrasts[-1]:.2f} (contrasts {contrasts})"
    )
