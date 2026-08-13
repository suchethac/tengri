# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the young-boundary knot captures recent star formation (#538).

The parametric SFH->SSP handoff samples the SFH at the SSP template ages, the
youngest of which is ~1 Myr. The mass formed between lookback 0 and that first
age was therefore never integrated into the youngest SSP bin, under-weighting
the ionizing population and biasing the ionizing-photon rate Q_H ~16 % low vs
the analytic (and CIGALE) SFH->SSP convolution -- n_ly drops 3+ dex past
~10 Myr, so the youngest bins dominate Q_H (#537, #538).

The fix prepends a lookback-0 knot to the DSPS SFH table (holding SFR constant
from the youngest sample), so DSPS integrates the youngest bin down to the
observation time. The knot REDISTRIBUTES mass into the youngest bin -- it does
not inflate the total (its [0, age0] segment is excluded from the normalization)
-- so ``sum(age_weights) == 10**log_total_mass`` is preserved.

These tests pin (1) the mechanism at the DSPS-table seam and (2) mass
conservation through the full component, both on the synthetic SSP (CI-safe).
The residual Q_H gap vs CIGALE (~6 % after this fix) is set by DSPS's
log-midpoint youngest-bin edge and is documented in #538.

References
----------
.. [1] S. Charlot, S. M. Fall, "A Simple Model for the Absorption of Starlight
   by Dust in Galaxies," ApJ, 539, 718 (2000).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.component import _build_dsps_sfh_table
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.regression_bug


def _synthetic_ssp():
    n_met, n_age, n_wave = 3, 20, 100
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, n_wave),
        ssp_flux=jnp.abs(jax.random.normal(jax.random.PRNGKey(123), (n_met, n_age, n_wave))) * 1e-3
        + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, n_age),
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


def test_young_knot_redistributes_into_youngest_bin():
    """The lookback-0 knot raises the youngest SSP-age weight (and conserves total).

    Run DSPS on the same delayed-tau SFH table with and without the knot; the
    youngest-bin probability weight must rise materially, while the totals stay
    equal (DSPS weights sum to one either way) -- i.e. the knot redistributes.
    """
    from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_lognormal_mdf

    ssp = _synthetic_ssp()
    ages_yr = (10.0 ** np.asarray(ssp.ssp_lg_age_gyr)) * 1e9
    # Young, actively star-forming delayed-tau galaxy (tau=0.5, age=1.0 Gyr).
    sfr = ages_yr * np.exp(-ages_yr / 0.5e9)
    t_obs_gyr = 13.0

    def youngest_weight(add_knot):
        t, s, _ = _build_dsps_sfh_table(
            jnp.asarray(ages_yr), jnp.asarray(sfr), t_obs_gyr, add_young_knot=add_knot
        )
        res = calc_rest_sed_sfh_table_lognormal_mdf(
            gal_t_table=t,
            gal_sfr_table=s,
            gal_lgmet=0.0,
            gal_lgmet_scatter=0.2,
            ssp_lgmet=ssp.ssp_lgmet,
            ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
            ssp_flux=ssp.ssp_flux,
            t_obs=t_obs_gyr,
        )
        w = np.asarray(res.weights)
        return float(w.sum(axis=0)[0]), float(w.sum())

    w0_off, tot_off = youngest_weight(False)
    w0_on, tot_on = youngest_weight(True)

    assert w0_on > 1.3 * w0_off, (
        f"young-boundary knot should raise the youngest-bin weight "
        f"(off={w0_off:.4e}, on={w0_on:.4e}); the recent-SF mass is being dropped"
    )
    # DSPS weights are probabilities -> both normalizations sum to one.
    assert abs(tot_off - 1.0) < 1e-4 and abs(tot_on - 1.0) < 1e-4


def test_young_knot_preserves_mass_conservation():
    """End-to-end: sum(age_weights) == 10**log_total_mass with the knot active."""
    ssp = _synthetic_ssp()
    for log_mass in (0.0, 9.5):
        m = SEDModel.build(
            ssp,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(0.5),
                "age_gyr": Fixed(1.0),
                "log_total_mass": Fixed(log_mass),
                "*": FIXED,
            },
            met={"logzsol": Fixed(0.0), "*": FIXED},
            dust={
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            redshift=Fixed(0.05),
        )
        aw = np.asarray(
            m.predict_state(dict(m.spec.sample(jax.random.PRNGKey(0)))).derived["age_weights"]
        )
        assert abs(aw.sum() / 10.0**log_mass - 1.0) < 1e-4, (
            f"mass not conserved with young-boundary knot: "
            f"sum(age_weights)={aw.sum():.6e} vs 10**{log_mass}"
        )
