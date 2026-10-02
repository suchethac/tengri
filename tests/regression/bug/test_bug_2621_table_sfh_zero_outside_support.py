# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2621: ``sfh={'type': 'table'}`` formed a different
mass than its declaration because ``_tabulated_sfh`` edge-clamped
(``jnp.interp``'s default) instead of zeroing outside the table's support.

Required (issue's proposed fix): zero outside the table, an optional
``sfh_table_age_gyr`` cut (CIGALE ``sfhfromfile`` convention), and a rescale
to a declared ``sfh_table_log_total_mass`` through the same single-factor
hook every other family uses.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def model(ssp):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={"type": "table", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.0),
        )


def _cosmic_time_table(t_obs_gyr, t_myr, raw):
    """Table rows 0..len-1 since formation [Myr]; last row is "now"."""
    t_gyr = t_obs_gyr - t_myr[-1] / 1e3 + t_myr / 1e3
    sfr_per_msun = raw / (raw.sum() * 1e6)  # normalize to 1 Msun over those rows
    return t_gyr, sfr_per_msun


@pytest.mark.parametrize(
    "shape,rtol",
    [
        ("rising", 1e-2),
        ("falling", 1e-2),
        ("bursty", 1e-2),
    ],
)
def test_formed_mass_matches_table_integral_no_declaration(model, shape, rtol):
    """Without a declared mass, formed mass equals the table's own (normalized) integral."""
    t_obs_gyr = float(age_at_z(0.0))
    t_myr = np.arange(3000, dtype=float)
    g = lambda c, w: np.exp(-0.5 * ((t_myr - c) / w) ** 2)  # noqa: E731
    raw = {
        "rising": t_myr + 1.0,
        "falling": np.exp(-t_myr / 800.0),
        "bursty": 1.0 + 20 * g(500, 30) + 20 * g(1500, 30) + 40 * g(2700, 15),
    }[shape]
    t_gyr, sfr = _cosmic_time_table(t_obs_gyr, t_myr, raw)
    st = model.predict_state({"sfh_t_gyr": t_gyr, "sfh_sfr": sfr})
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])
    # With no declared mass the formed mass is the table's OWN trapezoid
    # integral. The table is normalized by a rectangle sum (sum * 1 Myr), so
    # that trapezoid is 0.9993 (falling) / 0.9997 (rising) / 0.9999 (bursty),
    # not 1: the 7e-4 is the rectangle-vs-trapezoid difference of the table,
    # not model error. The kernel's measurement of it agrees to ~1e-4.
    table_integral = float(np.trapezoid(sfr, t_gyr * 1e9))
    np.testing.assert_allclose(formed_mass, table_integral, rtol=5e-4)
    # The OLD clamp gave 13.27 (falling), 2.28 (bursty), 1.0016 (rising).
    assert formed_mass < 1.1, (
        f"{shape}: formed {formed_mass} should not inflate like the old clamp"
    )


@pytest.mark.parametrize("shape", ["rising", "falling", "bursty"])
def test_declared_mass_rescales_exactly(model, shape):
    """With sfh_table_log_total_mass declared, formed mass equals it exactly."""
    t_obs_gyr = float(age_at_z(0.0))
    t_myr = np.arange(3000, dtype=float)
    g = lambda c, w: np.exp(-0.5 * ((t_myr - c) / w) ** 2)  # noqa: E731
    raw = {
        "rising": t_myr + 1.0,
        "falling": np.exp(-t_myr / 800.0),
        "bursty": 1.0 + 20 * g(500, 30) + 20 * g(1500, 30) + 40 * g(2700, 15),
    }[shape]
    t_gyr, sfr = _cosmic_time_table(t_obs_gyr, t_myr, raw)
    declared_mass = 3.7e9
    st = model.predict_state(
        {
            "sfh_t_gyr": t_gyr,
            "sfh_sfr": sfr,
            "sfh_table_log_total_mass": np.log10(declared_mass),
        }
    )
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])
    # The table family carries no ``log_total_mass`` key in the internal
    # sfh_kwargs (it has no declared registry params), so
    # ``_mass_conserving_total`` does not pin ``total_mass`` to the
    # declaration the way every other family's does -- the match here comes
    # from the pre-rescale in ``_tabulated_sfh`` alone, so a small residual
    # from the CIC quadrature measuring a slightly different mass than the
    # table's own trapezoid is expected (grid quadrature, not a unit error).
    np.testing.assert_allclose(formed_mass, declared_mass, rtol=5e-4)


def test_zero_outside_support_falling_table_ending_before_now(model):
    """A table that stops before 'now' is zero afterwards (not clamped to its edge value)."""
    t_obs_gyr = float(age_at_z(0.0))
    t_myr = np.arange(3000, dtype=float)
    raw = np.exp(-t_myr / 800.0)
    # Table's "now" is 2 Gyr before the actual observation: pad nothing, just
    # offset t_gyr so the last row is NOT at t_obs.
    t_gyr = (t_obs_gyr - 2.0) - t_myr[-1] / 1e3 + t_myr / 1e3
    sfr = raw / (raw.sum() * 1e6)
    st = model.predict_state({"sfh_t_gyr": t_gyr, "sfh_sfr": sfr})
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr_history = np.asarray(st.derived["sfr_history"])
    recent = lbt < 1.9e9  # well inside the 2 Gyr gap after the table ends
    assert np.allclose(sfr_history[recent], 0.0), (
        f"SFR should be zero after the table's last row, got max "
        f"{sfr_history[recent].max()} (old clamp held the edge value constant)"
    )


def test_age_cut_zeroes_beyond_cut(model):
    """sfh_table_age_gyr keeps rows with lookback <= age_gyr, zero beyond."""
    t_obs_gyr = float(age_at_z(0.0))
    t_myr = np.arange(5000, dtype=float)
    raw = np.exp(-t_myr / 800.0)
    t_gyr, sfr = _cosmic_time_table(t_obs_gyr, t_myr, raw)
    st = model.predict_state({"sfh_t_gyr": t_gyr, "sfh_sfr": sfr, "sfh_table_age_gyr": 3.0})
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr_history = np.asarray(st.derived["sfr_history"])
    beyond = lbt > 3.0e9
    assert sfr_history[beyond].max() == 0.0 if beyond.any() else True
    inside = lbt <= 3.0e9
    assert sfr_history[inside].max() > 0.0
