# SPDX-License-Identifier: BSD-3-Clause
"""#2640: every published SFR history is bounded to [0, age(z)] and carries the formed mass.

The stellar component's ``sfr_history`` and the ``SEDModel`` routes
(``predict_sfh``, ``predict_sfh_quantities``) publish ONE history: windowed to
the support by ``window_weight`` and rescaled by a single factor, so the PLAIN
integral over the published grid equals ``10**log_mstar_formed`` (float
precision on the native grid) and no star formation is published before the
Big Bang.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

Z = 0.1
LOG_MASS = 9.0


def _build(ssp, sfh):
    return SEDModel.build(
        ssp_data=ssp,
        sfh=sfh,
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(Z),
    )


def _cases():
    age = float(age_at_z(Z))
    for kernel in ("cic", "dsps"):
        yield pytest.param(
            {
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(LOG_MASS),
                "age_gyr": Fixed(1.1 * age),
                "age_kernel": kernel,
            },
            {},
            id=f"dpl-{kernel}",
        )
        yield pytest.param(
            {
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(LOG_MASS),
                "age_gyr": Fixed(1.1 * age),
                "age_kernel": kernel,
            },
            {},
            id=f"delayed-{kernel}",
        )
    xi = np.random.default_rng(3).standard_normal(256)
    yield pytest.param(
        {
            "type": ["dpl", "field"],
            "sfh_dpl_age_gyr": Fixed(1.1 * age),
            "sfh_dpl_log_total_mass": Fixed(LOG_MASS),
            "sfh_dpl_alpha": Fixed(1.0),
            "sfh_dpl_beta": Fixed(1.5),
            "sfh_dpl_tau_gyr": Fixed(3.0),
            "sfh_field_psd_sigma": Fixed(1.0),
            "sfh_field_psd_tau_myr": Fixed(50.0),
        },
        {"sfh_field_xi": xi},
        id="dpl+field-dsps",
    )
    for kernel in ("cic", "dsps"):
        t_myr = np.arange(3000.0)
        t_gyr = age - 3.0 + t_myr / 1e3
        sfr = np.exp(-t_myr / 800.0)
        sfr = sfr / (sfr.sum() * 1e6)
        yield pytest.param(
            {"type": "table", "all_params": Fixed(DEFAULT), "age_kernel": kernel},
            {"sfh_t_gyr": t_gyr, "sfh_sfr": sfr, "sfh_table_log_total_mass": LOG_MASS},
            id=f"table-{kernel}",
        )


@pytest.mark.parametrize(("sfh", "extra"), list(_cases()))
def test_published_histories_are_bounded_rescaled_and_agree(synthetic_ssp_wide, sfh, extra):
    model = _build(synthetic_ssp_wide, sfh)
    age_yr = float(age_at_z(Z)) * 1e9
    st = model.predict_state(extra)
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(st.derived["sfr_history"])
    formed = 10 ** float(st.derived["log_mstar_formed"])

    # No star formation published before the Big Bang (partial boundary cell
    # may straddle age(z); nothing strictly beyond the next node).
    beyond = lbt > age_yr * (1.0 + 1e-9)
    nxt = np.searchsorted(lbt, age_yr)  # first node beyond age(z)
    assert np.all(sfr[nxt + 1 :] == 0.0), "published SFR beyond age(z)"
    del beyond

    # The PLAIN integral equals the formed mass (window-weighted cell is
    # part of the published array, so no hard-cut boundary-cell error).
    np.testing.assert_allclose(np.trapezoid(sfr, lbt), formed, rtol=1e-8)

    # predict_sfh (native grid) is the same history, and its mean is bounded
    # and rescaled to the same mass.
    out = model.predict_sfh(extra, grid="native")
    np.testing.assert_allclose(np.asarray(out["sfr_full"]), sfr, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(np.trapezoid(np.asarray(out["sfr_mean"]), lbt), formed, rtol=1e-8)


@pytest.mark.parametrize(("sfh", "extra"), [c for c in _cases() if "table" not in c.id])
@pytest.mark.parametrize("n_linear", [200, 1000, 5000])
def test_resampled_linear_grid_is_bounded_and_carries_the_mass(
    synthetic_ssp_wide, sfh, extra, n_linear
):
    """``predict_sfh(grid='linear')``: exactly zero beyond age(z); integral ~ formed.

    The un-windowed rescaled history is resampled and the OUTPUT grid's own
    window is applied after, so no interpolation ramp crosses age(z). The
    integral differs from the formed mass only by the resampling error of the
    output grid (linear interpolation of the native history): < 1e-2 from
    1000 nodes, < 1e-1 for a bursty field sampled on 200.
    """
    model = _build(synthetic_ssp_wide, sfh)
    age_gyr = float(age_at_z(Z))
    st = model.predict_state(extra)
    formed = 10 ** float(st.derived["log_mstar_formed"])
    out = model.predict_sfh(extra, n_linear=n_linear)
    t = np.asarray(out["t_gyr"])
    for key in ("sfr_mean", "sfr_full"):
        s = np.asarray(out[key])
        assert t[s > 0].max() <= age_gyr, f"{key}: SFR > 0 beyond age(z)"
        np.testing.assert_allclose(
            np.trapezoid(s, t * 1e9), formed, rtol=1e-2 if n_linear >= 1000 else 1e-1
        )
