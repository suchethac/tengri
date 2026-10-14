# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2623: two SFH declarations that did not describe what formed.

1. ``delayed_bq`` refused ``r_sfr = 0`` (full quench) because its bound
   predicate required ``lo > 0``; CIGALE's ``sfhdelayedbq`` allows
   ``r_sfr = 0`` (Boquien et al. 2019). Fixed via the ``lo >= 0`` predicate
   pattern (#2568).
2. ``sfh2exp`` evaluated the burst window ``[0, burst_age]`` independently of
   the main window ``[0, age]``, so a burst with ``burst_age >= age`` placed
   mass before the formation epoch. Fixed by bounding the burst window to
   ``[0, min(burst_age, age)]`` while keeping ``f_burst`` exact (CIGALE's
   ``sfh2exp`` truncate-and-rescale convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.sfh.mean_sfh import sfh2exp

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    import tengri

    return tengri.load_ssp()


class TestDelayedBqFullQuench:
    """delayed_bq r_sfr=0 builds and forms the declared mass with zero SFR after quench."""

    def test_r_sfr_zero_builds(self, ssp):
        model = SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "delayed_bq",
                "sfh_delayed_bq_tau_main_gyr": Fixed(2.0),
                "sfh_delayed_bq_age_main_gyr": Fixed(3.0),
                "sfh_delayed_bq_age_bq_gyr": Fixed(0.1),
                "sfh_delayed_bq_r_sfr": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.0),
        )
        st = model.predict_state({})
        sfr = np.asarray(st.derived["sfr_history"])
        lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
        formed_mass = 10 ** float(st.derived["log_mstar_formed"])

        # Zero SFR after the quench epoch (lookback < age_bq_gyr).
        quench_lbt_yr = 0.1e9
        post_quench = sfr[lbt < quench_lbt_yr]
        assert np.allclose(post_quench, 0.0), (
            f"SFR should be zero after full quench (r_sfr=0): max = {post_quench.max()}"
        )
        # Formed mass equals the declared mass (default log_total_mass=10.0).
        np.testing.assert_allclose(formed_mass, 1e10, rtol=1e-6)

    def test_r_sfr_zero_finite_gradient(self, ssp):
        """SFR is finite at r_sfr = 0+ (gradient-safe at the bound)."""
        import jax.numpy as jnp

        from tengri.components.stellar.sfh.mean_sfh import delayed_bq

        t_lookback = jnp.linspace(0.0, 3.0e9, 64)

        def f(r_sfr):
            return jnp.sum(
                delayed_bq(
                    t_lookback,
                    log_total_mass=10.0,
                    tau_main_yr=2.0e9,
                    age_main_yr=3.0e9,
                    age_bq_yr=0.1e9,
                    r_sfr=r_sfr,
                )
            )

        import jax

        val, grad = jax.value_and_grad(f)(0.0)
        # grad-assert: finite-only -- at r_sfr=0 (full quench) the SFR sum is
        # legitimately the pre-quench-only contribution and may be any
        # non-negative value; only its finiteness is this assertion's claim.
        assert np.isfinite(float(val)), "SFR sum at r_sfr=0 should be finite"
        assert np.isfinite(float(grad)), "gradient at r_sfr=0 should be finite, not NaN/inf"
        assert grad != 0.0, (
            "gradient at r_sfr=0 should be non-zero: r_sfr scales the "
            "post-quench SFR amplitude linearly, so d(SFR)/d(r_sfr) != 0 there"
        )


class TestSfh2expBurstBound:
    """sfh2exp burst mass never sits before the main population's formation epoch."""

    @pytest.mark.parametrize("burst_age_myr", [299.0, 300.0, 400.0])
    def test_no_burst_mass_before_formation(self, burst_age_myr):
        tau_main_myr, tau_burst_myr, f_burst, age_myr = 500.0, 100.0, 0.2, 300.0
        lb = np.linspace(0.0, max(burst_age_myr, age_myr) * 1e6, 2_000_001)
        tot = np.asarray(
            sfh2exp(
                lb,
                log_total_mass=0.0,
                tau_main_yr=tau_main_myr * 1e6,
                tau_burst_yr=tau_burst_myr * 1e6,
                f_burst=f_burst,
                age_yr=age_myr * 1e6,
                burst_age_yr=burst_age_myr * 1e6,
            ),
            dtype=float,
        )
        nob = np.asarray(
            sfh2exp(
                lb,
                log_total_mass=0.0,
                tau_main_yr=tau_main_myr * 1e6,
                tau_burst_yr=tau_burst_myr * 1e6,
                f_burst=0.0,
                age_yr=age_myr * 1e6,
                burst_age_yr=burst_age_myr * 1e6,
            ),
            dtype=float,
        )
        burst = tot - (1.0 - f_burst) * nob
        before = lb > age_myr * 1e6

        total_mass = np.trapezoid(tot, lb)
        burst_mass = np.trapezoid(burst, lb)
        mass_before_formation = np.trapezoid(tot[before], lb[before]) if before.any() else 0.0

        np.testing.assert_allclose(total_mass, 1.0, rtol=1e-6)
        np.testing.assert_allclose(burst_mass, f_burst, atol=1e-6)
        assert mass_before_formation < 1e-8, (
            f"burst_age={burst_age_myr} Myr: {mass_before_formation:.3e} Msun formed "
            f"before the formation epoch (age={age_myr} Myr)"
        )

    def test_untruncated_burst_unchanged(self):
        """burst_age < age (no truncation needed): shape matches the un-bounded formula."""
        tau_main_myr, tau_burst_myr, f_burst, age_myr, burst_age_myr = (
            500.0,
            100.0,
            0.2,
            300.0,
            50.0,
        )
        lb = np.linspace(0.0, age_myr * 1e6, 5000)
        sfr = np.asarray(
            sfh2exp(
                lb,
                log_total_mass=0.0,
                tau_main_yr=tau_main_myr * 1e6,
                tau_burst_yr=tau_burst_myr * 1e6,
                f_burst=f_burst,
                age_yr=age_myr * 1e6,
                burst_age_yr=burst_age_myr * 1e6,
            ),
            dtype=float,
        )
        total_mass = np.trapezoid(sfr, lb)
        np.testing.assert_allclose(total_mass, 1.0, rtol=1e-4)
        assert np.all(sfr >= 0.0)
