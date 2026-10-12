# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2635: the [0, age0] sliver (younger than any SSP
template) must carry the SFH's TRUE mass there, on both age kernels.

Before this fix, the age kernel approximated the sliver as a rectangle of
height ``SFR(age0)`` and width ``age0`` (the first-sample value held
constant). An SFH that ends before ``age0`` has ``SFR(age0) = 0``, so the
whole sliver -- and, for an SFH wholly inside it, the whole population --
vanished with no error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def age0_yr(ssp):
    ages = (10.0 ** np.asarray(ssp.ssp_lg_age_gyr)) * 1e9
    return float(ages[0])


def _build_const(ssp, age_kernel, start_gyr, end_gyr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "const",
                "start_gyr": Fixed(start_gyr),
                "end_gyr": Fixed(end_gyr),
                "log_total_mass": Fixed(9.0),
                "age_kernel": age_kernel,
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.0),
        )


@pytest.mark.parametrize("age_kernel", ["cic", "dsps"])
def test_sfh_wholly_inside_sliver_forms_declared_mass(ssp, age0_yr, age_kernel):
    """A burst that starts AND ends before age0 forms the full declared mass."""
    model = _build_const(
        ssp, age_kernel, start_gyr=age0_yr * 0.9 / 1e9, end_gyr=age0_yr * 0.1 / 1e9
    )
    st = model.predict_state({})
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])
    np.testing.assert_allclose(
        formed_mass,
        1e9,
        rtol=1e-6,
        err_msg=f"{age_kernel}: wholly inside [0,age0] formed {formed_mass:.3e}, declared 1e9",
    )


@pytest.mark.parametrize("age_kernel", ["cic", "dsps"])
def test_sfh_partly_inside_sliver_not_undercounted(ssp, age0_yr, age_kernel):
    """A burst straddling age0 (lower SFR near age0) still forms the declared mass."""
    model = _build_const(
        ssp, age_kernel, start_gyr=age0_yr * 3.0 / 1e9, end_gyr=age0_yr * 0.3 / 1e9
    )
    st = model.predict_state({})
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])
    np.testing.assert_allclose(formed_mass, 1e9, rtol=1e-6)


@pytest.mark.parametrize("age_kernel", ["cic", "dsps"])
def test_smooth_control_unchanged(ssp, age0_yr, age_kernel):
    """A control SFH that extends well past age0 forms the declared mass to 1e-4 (#2635 pin)."""
    model = _build_const(ssp, age_kernel, start_gyr=5.0, end_gyr=0.0001)
    st = model.predict_state({})
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])
    np.testing.assert_allclose(formed_mass, 1e9, rtol=1e-4)
