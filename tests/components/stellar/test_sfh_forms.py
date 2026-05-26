# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parametric SFH form validity and cumulative mass bounds.

Every parametric SFH form must produce a non-negative SFR array with realistic cumulative mass.
"""

from __future__ import annotations

import chex
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


def _make_ssp_if_available():
    from pathlib import Path

    from tengri import load_ssp_data

    p = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    if not p.exists():
        return None
    return load_ssp_data(str(p))


class TestSFHForms:
    """Every parametric SFH form must produce a non-negative SFR array of the
    right shape, with cumulative stellar mass ~ 10^7–10^11 M⊙ for SFR ~ 1."""

    @pytest.mark.parametrize(
        "sfh_type,kwargs",
        [
            (
                "dpl",
                {
                    "sfh_dpl_log_total_mass": 0.0,
                    "sfh_dpl_alpha": 2.0,
                    "sfh_dpl_beta": 1.0,
                    "sfh_dpl_tau_gyr": 3.0,
                },
            ),
            (
                "lnorm",
                {
                    "sfh_lnorm_log_total_mass": 0.0,
                    "sfh_lnorm_peak_lbt_gyr": 3.0,
                    "sfh_lnorm_width_gyr": 1.0,
                },
            ),
            (
                "tsnorm",
                {
                    "sfh_tsnorm_log_total_mass": 0.0,
                    "sfh_tsnorm_peak_lbt_gyr": 3.0,
                    "sfh_tsnorm_width_gyr": 2.0,
                    "sfh_tsnorm_skew": 0.0,
                    "sfh_tsnorm_trunc": 5.0,
                },
            ),
        ],
    )
    def test_sfh_produces_valid_array(self, sfh_type, kwargs):
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")
        param_dict = {k: Fixed(v) for k, v in kwargs.items()}
        param_dict.update(
            {
                "met_logzsol": Fixed(0.0),
                "dust_tau_bc": Fixed(0.0),
                "dust_tau_diff": Fixed(0.0),
                "dust_slope": Fixed(-0.7),
                "redshift": Fixed(0.0),
                "mean_sfh_type": sfh_type,
            }
        )
        spec = Parameters(**param_dict)
        model = SEDModel(spec, ssp)
        sfh = model.predict_sfh({})
        sfr = np.array(sfh["sfr_mean"])
        t = np.array(sfh["t_gyr"])
        chex.assert_equal_shape([sfr, t])
        assert np.all(sfr >= 0.0), f"{sfh_type}: negative SFR"
        chex.assert_tree_all_finite(sfr)
        dt_yr = np.abs(np.diff(t)) * 1e9
        mass = float(np.sum(sfr[:-1] * dt_yr))
        assert 1e7 < mass < 1e12, f"{sfh_type}: cumulative mass {mass:.2e} M⊙ outside [1e7, 1e12]"
