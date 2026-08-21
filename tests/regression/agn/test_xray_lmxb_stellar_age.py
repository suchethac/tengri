# SPDX-License-Identifier: BSD-3-Clause
"""Regression: X-ray LMXB scaling must use the galaxy's mass-weighted age.

Building the CIGALE panchromatic head-to-head
(``reproduction/cigale/01_cigale.py``) exposed a wiring bug: the
``XRaySEDComponent`` never passed ``stellar_age_gyr`` to ``xray_total``, so the
Lehmer+2016 **LMXB** term — which dominates the galaxy X-ray — used the 1 Gyr
default regardless of the stellar population. The ``logT`` polynomial is steep,
so an evolved (~3 Gyr) galaxy came out ~3-4x too luminous vs CIGALE ``yang20``
(which uses ``stellar.age_m_star``). The component now threads the SSP
mass-weighted age; this test guards that wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.xray.xray import xray_total

pytestmark = [pytest.mark.regression_bug]


def _mass_weighted_age_gyr(derived) -> float:
    aw = np.asarray(derived["age_weights"])
    ages = np.asarray(derived["ssp_ages_yr"])
    return float((aw * ages).sum() / aw.sum() / 1.0e9)


def test_lmxb_uses_mass_weighted_age(synthetic_ssp_wide) -> None:
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        met={"*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(8.0),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.3),
            "*": FIXED,
        },
        dust_emission={"type": "dale2014", "*": FIXED},
        xray={"type": "yang20", "*": FIXED},
        redshift=Fixed(0.0),
    )
    state = model.predict_state({})
    derived = state.derived
    mwa = _mass_weighted_age_gyr(derived)

    wave = np.asarray(state.wave)
    sed_xray = np.asarray(derived["sed_xray"])
    sfr = float(derived["sfr"])
    mstar = 10.0 ** float(derived["log_mstar"])

    # The component's X-ray must reproduce ``xray_total`` evaluated at the
    # mass-weighted age (the wiring under test), not at the 1 Gyr default.
    lx_mwa = np.asarray(
        xray_total(wave, sfr=sfr, stellar_mass=mstar, l_2500_30deg=0.0, stellar_age_gyr=mwa)
    )
    np.testing.assert_allclose(sed_xray, lx_mwa, rtol=0.02, atol=0.0)

    # And with an evolved population (mass-weighted age well above 1 Gyr) the
    # LMXB-dominated hard band must sit *below* the buggy 1 Gyr-default value —
    # this is the direction of the fix and guards against a silent revert.
    lx_1gyr = np.asarray(
        xray_total(wave, sfr=sfr, stellar_mass=mstar, l_2500_30deg=0.0, stellar_age_gyr=1.0)
    )
    j_3kev = int(np.argmin(np.abs(wave - 12.398 / 3.0)))  # ~3 keV, LMXB-dominated
    assert mwa > 1.5, f"expected an evolved population; mass-weighted age = {mwa:.2f} Gyr"
    assert sed_xray[j_3kev] < 0.7 * lx_1gyr[j_3kev], (
        f"X-ray at 3 keV ({sed_xray[j_3kev]:.3e}) is not below the 1 Gyr-default "
        f"value ({lx_1gyr[j_3kev]:.3e}); the LMXB age wiring has regressed"
    )
