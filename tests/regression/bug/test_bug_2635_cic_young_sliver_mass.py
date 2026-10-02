# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2635: the [0, age0] sliver (younger than any SSP
template) must carry the SFH's TRUE mass there, on both age kernels.

Before this fix, both kernels approximated the sliver as a rectangle of
height ``SFR(age0)`` and width ``age0`` (the #538 young-boundary knot held
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


def test_dsps_knot_clamp_overcount_is_bounded():
    """``_sliver_equivalent_sfr0`` clamps at 0 when the SFH quenches inside the sliver.

    SFR = 1 on lookback [0.8 age0, inf) and 0 inside [0, 0.8 age0) -- a quench
    in the last age0 years. The true sliver mass is 0.2 * age0 (SFR 1 only on
    [0.8, 1] age0); the one-knot cell cannot carry a negative SFR, so it carries
    SFR(age0) * age0 / 2 = 0.5 * age0: an over-count of 0.3 * age0, bounded by
    the documented SFR(age0) * age0 / 2 (and below the 0.8 * age0 of a
    rectangle held at SFR(age0)).
    """
    import jax.numpy as jnp

    from tengri.components.stellar.component import _sliver_equivalent_sfr0

    age0 = 3.0e5
    ssp_ages = jnp.asarray([age0, 2 * age0, 4 * age0])

    def sfh(t, **_):
        return jnp.where(t >= 0.8 * age0, 1.0, 0.0)

    v0 = float(_sliver_equivalent_sfr0(ssp_ages, sfh, {}))
    assert v0 == 0.0  # clamped
    cell_mass = 0.5 * (v0 + 1.0) * age0
    true_mass = 0.2 * age0
    overcount = cell_mass - true_mass
    assert overcount == pytest.approx(0.3 * age0, rel=0.05)
    assert overcount <= 0.5 * 1.0 * age0
    assert overcount < 0.8 * age0


def test_dsps_knot_clamp_overcount_limit_history_vanishing_in_sliver():
    """Limit M_sliver -> 0: the over-count reaches, and does not exceed, its bound.

    SFR = 1 for lookback >= age0 and 0 strictly inside the sliver, so the true
    sliver mass is 0 and the one-knot cell (clamped at v0 = 0) carries
    SFR(age0) * age0 / 2 -- exactly the documented bound.
    """
    import jax.numpy as jnp

    from tengri.components.stellar.component import _sliver_equivalent_sfr0

    age0 = 3.0e5
    ssp_ages = jnp.asarray([age0, 2 * age0, 4 * age0])

    def sfh(t, **_):
        return jnp.where(t >= age0, 1.0, 0.0)

    v0 = float(_sliver_equivalent_sfr0(ssp_ages, sfh, {}))
    assert v0 == 0.0
    overcount = 0.5 * (v0 + 1.0) * age0 - 0.0
    assert overcount == pytest.approx(0.5 * age0, rel=1e-6)
