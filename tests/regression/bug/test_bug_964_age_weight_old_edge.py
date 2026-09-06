# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #964 — SSP age weights vs a dense code-independent reference.

Bug: the delta-metallicity path fed DSPS a coarse per-SSP-age SFH table, and
DSPS's histogram kernel interpolates log10(cumulative mass) in log10(t) —
across the table segment straddling the SFH's maximum age the geometric
interpolation annihilates that segment's mass, so the first SSP node older
than the SFH start received exactly ZERO weight (3.8 % of all mass for a
delayed-τ with age = 5 Gyr on the MIST grid). Renormalization pushed the
missing mass onto younger, brighter nodes: a +1.2 % optical CSP bias vs
FSPS, bagpipes (#499), and the dense reference — with a blue-ward tilt
(#858's UV excess is the same family).

Fix: tengri computes the age marginal itself with a cloud-in-cell kernel
(``_age_weights_cic``) on the dense integrand — each mass parcel splits
between its bracketing SSP nodes with log-age interpolation weights, the
convention FSPS matches to 1e-4. These tests pin the weights against an
independent dense-quadrature reference on the synthetic SSP grid.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug


def _reference_weights(sfr_fn, lg_age_gyr, n=1_000_000):
    """Dense CIC quadrature: split SFR(t)·dt parcels between bracketing nodes."""
    age_yr = 10.0 ** np.asarray(lg_age_gyr) * 1e9
    t = np.geomspace(1e3, age_yr[-1], n)
    sfr = sfr_fn(t)
    x_t = np.log10(t / 1e9)
    idx = np.clip(np.searchsorted(lg_age_gyr, x_t) - 1, 0, len(lg_age_gyr) - 2)
    f = np.clip((x_t - lg_age_gyr[idx]) / (lg_age_gyr[idx + 1] - lg_age_gyr[idx]), 0.0, 1.0)
    dt = np.empty_like(t)
    dt[1:-1] = 0.5 * (t[2:] - t[:-2])
    dt[0] = 0.5 * (t[1] - t[0])
    dt[-1] = 0.5 * (t[-1] - t[-2])
    contrib = sfr * dt
    w = np.zeros(len(lg_age_gyr))
    np.add.at(w, idx, contrib * (1.0 - f))
    np.add.at(w, idx + 1, contrib * f)
    return w / w.sum()


def _age_marginal(ssp, sfh):
    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh=sfh,
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    jw = np.asarray(s.derived["joint_weights"])
    return jw.sum(axis=0), s


class TestAgeWeightOldEdge:
    def test_delayed_tau_matches_dense_reference(self, synthetic_ssp_wide):
        """Per-node weights track the dense reference; no annihilated node."""
        ssp = synthetic_ssp_wide
        lg_age = np.asarray(ssp.ssp_lg_age_gyr)
        tau_yr, age_yr_max = 1e9, 5.0e9  # SFH start deliberately off-grid

        def sfr_delayed(t_lb):
            ts = age_yr_max - t_lb
            shape = np.where(ts > 0, ts * np.exp(-ts / tau_yr), 0.0)
            return shape

        w_t, _ = _age_marginal(
            ssp,
            {
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
        )
        w_r = _reference_weights(sfr_delayed, lg_age)
        assert np.abs(w_t - w_r).sum() < 2e-3, f"L1(weights) = {np.abs(w_t - w_r).sum():.2e}"

        # The smoking gun: the first node OLDER than the SFH's maximum age
        # must carry its interpolation share (the old kernel gave exactly 0).
        i_above = int(np.searchsorted(10.0**lg_age * 1e9, age_yr_max))
        assert w_r[i_above] > 0.01  # the reference puts real mass there
        assert w_t[i_above] == pytest.approx(w_r[i_above], rel=0.05)

    def test_young_tophat_matches_dense_reference(self, synthetic_ssp_wide):
        """A young constant-SFR bin (continuity, one dominant bin) keeps its old slice."""
        ssp = synthetic_ssp_wide
        lg_age = np.asarray(ssp.ssp_lg_age_gyr)

        # A top-hat over the youngest default bin, [0, 30 Myr]. `dirichlet` was
        # the vehicle here, via a helper that inverted tengri's stick-breaking
        # on the *mass* fractions. That inverse encoded the pre-Leja+2017
        # mapping (the stick-breaking now runs on the SFR fractions, from
        # Beta(1, N-1-i)-mapped latents), so the test would have to restate
        # the prior's transform to keep using it. `continuity` reaches the
        # same SFH directly: one large log-SFR ratio puts 1e9x the SFR in the
        # youngest bin, i.e. all but 5e-7 of the mass.
        sfh = {
            "type": "continuity",
            "log_total_mass": Fixed(10.0),
            "ratio_0": Fixed(9.0),
            "all_params": Fixed(DEFAULT),
        }
        w_t, _ = _age_marginal(ssp, sfh)

        def sfr_top(t_lb):  # tengri's bin convention: uniform over [0, 30 Myr]
            return np.where(t_lb <= 30e6, 1.0, 0.0)

        w_r = _reference_weights(sfr_top, lg_age)
        assert np.abs(w_t - w_r).sum() < 2e-3, f"L1(weights) = {np.abs(w_t - w_r).sum():.2e}"

    def test_mass_conservation(self, synthetic_ssp_wide):
        """sum(age_weights) must equal 10**log_total_mass exactly."""
        _, s = _age_marginal(
            synthetic_ssp_wide,
            {
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
        )
        total = float(jnp.sum(jnp.asarray(s.derived["age_weights"])))
        assert total == pytest.approx(1e10, rel=1e-6)
