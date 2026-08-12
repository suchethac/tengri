# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parametric double-power-law SFH model (DPL)."""

import chex
import jax.numpy as jnp
import numpy as np
import pytest


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


pytestmark = pytest.mark.bounds

AGE_YR = jnp.linspace(1e7, 13.5e9, 200)


class TestSFHForms:
    """Every parametric SFH form must produce a non-negative SFR array of the
    right shape, with cumulative stellar mass ~ 10^7–10^11 M⊙ for SFR ~ 1."""

    @pytest.mark.parametrize(
        "sfh_type,kwargs",
        [
            (
                "dpl",
                {
                    # log_total_mass is log10(M⊙) under the total_mass contract
                    # (trapezoid(sfr,t)=10^log_total_mass), so 9.0 → ~1e9 M⊙,
                    # inside the [1e7,1e12] sanity band (0.0 → 1 M⊙ was stale).
                    "sfh_dpl_log_total_mass": 9.0,
                    # Formation anchor (cosmic time for SF) — a free param since
                    # #549; fix it for this all-Fixed predict_sfh({}) check.
                    # z=0 source ⇒ age_of_universe(0); ≤ _AGE_UNIV_GYR ≈ 13.81.
                    "sfh_dpl_age_gyr": 13.0,
                    "sfh_dpl_alpha": 2.0,
                    "sfh_dpl_beta": 1.0,
                    "sfh_dpl_tau_gyr": 3.0,
                },
            ),
            (
                "lnorm",
                {
                    "sfh_lnorm_log_total_mass": 9.0,
                    "sfh_lnorm_age_gyr": 13.0,  # formation anchor (free since #549)
                    "sfh_lnorm_peak_gyr": 3.0,
                    "sfh_lnorm_width_gyr": 1.0,
                },
            ),
            (
                "tsnorm",
                {
                    "sfh_tsnorm_log_total_mass": 9.0,
                    "sfh_tsnorm_peak_lbt_gyr": 3.0,
                    "sfh_tsnorm_width_gyr": 2.0,
                    "sfh_tsnorm_skew": 0.0,
                    "sfh_tsnorm_trunc": 5.0,
                },
            ),
        ],
    )
    def test_sfh_produces_valid_array(self, sfh_type, kwargs, synthetic_ssp_wide):
        from tengri import Fixed, Parameters, SEDModel

        # #613: predict_sfh is SSP-agnostic (uses only the age grid), so the
        # shared synthetic SSP lets this SFH-shape check run on CI.
        ssp = synthetic_ssp_wide
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
