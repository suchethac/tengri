# SPDX-License-Identifier: BSD-3-Clause
"""A tabulated SFH with the GP field (#3505, row B25).

The table fixes the mean star-formation history. The GP field multiplies the table
SFR by the lognormal modulation ``exp(x(t) - K(0)/2)``, and the formed mass varies
from draw to draw, with its expectation equal to the table integral (owner ruling,
2026-11-15). Four things are pinned here:

* the model builds and runs, where it used to raise ``NotImplementedError``;
* at ``xi = 0`` the history is the table times the constant ``exp(-K(0)/2)``;
* the mean formed mass over many ``xi`` draws matches the table integral within a
  Monte-Carlo tolerance;
* the formed mass is not pinned per draw, so its spread is non-zero.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import StellarSEDComponent
from tengri.components.stellar.sfh.registry import compute_field_gp
from tengri.cosmology import age_at_z
from tengri.parameters.resolve import merge_fixed_params

pytestmark = pytest.mark.regression_bug

#: Cosmic-time nodes of the table [Gyr]: a burst-and-decline history wholly inside
#: the universe at z = 0.1 (age ~12.9 Gyr), so no node is cut at the age of the universe.
T_NODES_GYR = np.linspace(1.0, 10.0, 19)
SFR_NODES = 2.0 + 3.0 * np.exp(-0.5 * ((T_NODES_GYR - 4.0) / 1.5) ** 2)
Z_OBS = 0.1
N_GRID = 64
PSD_SIGMA = 0.3
PSD_TAU_MYR = 200.0
#: Dummy SSP age grid: the field is evaluated on the lookback grid, not on these.
SSP_AGES_YR = 10.0 ** np.linspace(6.0, 10.1, 12)


def _build(ssp, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": ["table", "field"], "all_params": Fixed(DEFAULT)},
            dust_attenuation=None,
            neb={"type": "none"},
            redshift=Fixed(Z_OBS),
            n_grid=N_GRID,
        )


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return _build(synthetic_ssp_wide, synthetic_tophat_obs)


@pytest.fixture(scope="module")
def stellar(model):
    chain = model._build_component_chain()
    return next(c for c in chain if isinstance(c, StellarSEDComponent))


def _params(model, xi, sigma=PSD_SIGMA):
    """Full parameter dict: the table arrays, the field knobs and one latent draw."""
    spec = model.spec
    params = merge_fixed_params(spec, dict(spec.sample(jax.random.PRNGKey(0))))
    params["sfh_t_gyr"] = jnp.asarray(T_NODES_GYR)
    params["sfh_sfr"] = jnp.asarray(SFR_NODES)
    params["sfh_field_psd_sigma"] = jnp.asarray(sigma)
    params["sfh_field_psd_tau_myr"] = jnp.asarray(PSD_TAU_MYR)
    params["sfh_field_xi"] = jnp.asarray(xi)
    return params


def _table_integral(t_obs_gyr):
    """Trapezoid of the table over its lookback support: the target of the field mean."""
    lbt_yr = (t_obs_gyr - T_NODES_GYR[::-1]) * 1e9
    return float(np.trapezoid(SFR_NODES[::-1], lbt_yr))


def test_table_field_builds_and_runs(model, stellar):
    """The table with the field builds and forms positive, finite mass (was a raise)."""
    rng = np.random.default_rng(1)
    _, mass, _ = stellar.compute_joint_weights(_params(model, rng.standard_normal(N_GRID)))
    assert np.isfinite(float(mass)) and float(mass) > 0.0


def test_table_field_apply_runs(model):
    """The exact forward (apply) builds and predicts finite photometry too."""
    rng = np.random.default_rng(2)
    params = _params(model, rng.standard_normal(N_GRID))
    free_only = {k: v for k, v in params.items() if k not in set(model.spec.fixed_params)}
    flux = np.asarray(model.predict_photometry(free_only))
    assert np.all(np.isfinite(flux)) and np.all(flux >= 0.0)


def test_xi_zero_is_the_table_times_the_lognormal_constant(model, stellar):
    """At xi = 0 the SFR is the table times exp(-K(0)/2): one constant on every node."""
    params = _params(model, np.zeros(N_GRID))
    out = stellar._sfh_on_ssp_ages(params, jnp.asarray(SSP_AGES_YR), N_GRID)
    lbt_yr = np.asarray(out.sfh_lbt_grid)
    table = np.interp(
        lbt_yr,
        (age_at_z(Z_OBS) - T_NODES_GYR[::-1]) * 1e9,
        SFR_NODES[::-1],
        left=0.0,
        right=0.0,
    )
    _, k0_half = compute_field_gp(
        jnp.zeros(N_GRID),
        jnp.asarray(PSD_SIGMA),
        jnp.asarray(PSD_TAU_MYR * 1e6),
        N_GRID,
        float(np.log10(lbt_yr[1]) - np.log10(lbt_yr[0])),
        log_age_grid=jnp.log10(jnp.asarray(lbt_yr)),
    )
    inside = table > 0.0
    # The table spans 2.9-11.9 Gyr of lookback, a handful of the log-age nodes.
    assert inside.sum() >= 5, "the table must cover several lookback grid nodes"
    sfr = np.asarray(out.sfr_history)
    np.testing.assert_allclose(sfr[inside] / table[inside], np.exp(-float(k0_half)), rtol=1e-8)
    np.testing.assert_array_equal(sfr[~inside], 0.0)


def test_mean_formed_mass_over_draws_equals_the_table_integral(model, stellar):
    """E[formed mass] over the latent draws is the table integral, within Monte-Carlo error.

    The lognormal correction makes E[exp(x - K/2)] = 1 at each node, so the expected SFR
    is the table and the expected mass is its integral. The tolerance is four standard
    errors of the draw mean plus 2 % for the finite lookback grid.
    """
    target = _table_integral(age_at_z(Z_OBS))
    rng = np.random.default_rng(11)
    masses = np.array(
        [
            float(stellar.compute_joint_weights(_params(model, rng.standard_normal(N_GRID)))[1])
            for _ in range(200)
        ]
    )
    mean = masses.mean()
    se = masses.std(ddof=1) / np.sqrt(masses.size)
    assert abs(mean - target) <= 4.0 * se + 0.02 * target, (
        f"mean formed mass {mean:.4e} vs table integral {target:.4e} (SE {se:.2e})"
    )


def test_formed_mass_varies_per_draw(model, stellar):
    """The table pins the MEAN mass, not each draw: the spread across draws is non-zero."""
    rng = np.random.default_rng(5)
    masses = np.array(
        [
            float(stellar.compute_joint_weights(_params(model, rng.standard_normal(N_GRID)))[1])
            for _ in range(20)
        ]
    )
    assert np.std(masses) > 1e-3 * np.mean(masses)
