# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the energy-balance B/G tables read metallicity by the native-slope Hermite.

The stellar absorbed luminosity is read per SSP across metallicity. The B and G families
were PCHIP-read over their own integrated values, which is a different number because PCHIP
is nonlinear in the data. The fix carries native PCHIP slopes through the same linear B/G
build and reads them by Hermite (owner ruling 2026-11-15, "Exact: native slopes").

The check is the dust emission contribution to two WISE bands (W3, W4) of a two-component
calzetti screen with a dale2014 emitter, from the emitter on minus off, at an off-node
metallicity. The LUT contribution must match the exact one. Before the fix the relative gap
on main was 9.3e-5 per band.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp
from tests.contract import _band_table_cases as cases

pytestmark = pytest.mark.regression_bug

#: Relative tolerance on the emitter's contribution: the measured gap after the fix is 1.1e-8.
CONTRIBUTION_RTOL = 1e-7


def _model(approx, emitter):
    obs = Observation(
        photometry=Photometry.from_names(["wise_w3", "wise_w4"], convention="bessell")
    )
    return SEDModel.build(
        ssp_data=cases._bare_ssp(),
        observation=obs,
        approx=approx,
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        dust_emission=(
            {"type": "dale2014", "all_params": Fixed(DEFAULT)} if emitter else {"type": "none"}
        ),
    )


def _contribution(approx):
    on = np.asarray(_model(approx, True).predict_photometry({}), dtype=np.float64)
    off = np.asarray(_model(approx, False).predict_photometry({}), dtype=np.float64)
    return on - off


def test_dust_emission_contribution_lut_matches_exact():
    exact = _contribution(None)
    lut = _contribution(WavePrecomp())
    rel = np.abs(lut / exact - 1.0)
    assert float(np.max(rel)) <= CONTRIBUTION_RTOL, rel
