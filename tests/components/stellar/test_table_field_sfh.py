# SPDX-License-Identifier: BSD-3-Clause
"""A tabulated SFH with the GP field (#3505).

The GP field multiplies the table SFR by the lognormal modulation
``exp(x(t) - K(0)/2)``, which changes the SFR shape draw by draw. As for a parametric
SFH under the field (#2622), every draw's formed mass is pinned to the table integral
(the declared mass is each draw's own, #2622). Pinned here:

* the model builds and runs, where it used to raise ``NotImplementedError``;
* the exact forward (apply) predicts finite, non-negative photometry;
* at ``xi = 0`` the history is the table times the constant ``exp(-K(0)/2)``;
* every ``xi`` draw forms the table integral, to float precision, on both the exact
  path and the fast path (the same pin, per draw);
* the field still changes the SFR shape: at some ages the draw differs from the table
  by more than 1 % in shape.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import StellarSEDComponent, evaluate_sfh
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


def _table_on(lbt_yr):
    return np.interp(
        lbt_yr, (age_at_z(Z_OBS) - T_NODES_GYR[::-1]) * 1e9, SFR_NODES[::-1], left=0.0, right=0.0
    )


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
    out = evaluate_sfh(stellar.config, params, jnp.asarray(SSP_AGES_YR), N_GRID)
    lbt_yr = np.asarray(out.sfh_lbt_grid)
    table = _table_on(lbt_yr)
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


def test_every_draw_forms_the_table_integral_on_both_paths(model, stellar):
    """Each draw's formed mass is the table integral, on apply and on the fast path.

    The per-draw pin of #2622 for parametric SFHs, carried to the table: the field
    changes each draw's shape, not its formed mass.
    """
    target = _table_integral(age_at_z(Z_OBS))
    rng = np.random.default_rng(11)
    for _ in range(8):
        xi = rng.standard_normal(N_GRID)
        params = _params(model, xi)
        fast_mass = float(stellar.compute_joint_weights(params)[1])
        assert fast_mass == pytest.approx(target, rel=1e-10), (
            f"fast path formed {fast_mass:.10e}, table {target:.10e}"
        )
        state = stellar.apply(_forward_state(), params, stellar.ssp_data)
        exact_mass = 10.0 ** float(state.derived["log_mstar_formed"])
        assert exact_mass == pytest.approx(target, rel=1e-10), (
            f"exact path formed {exact_mass:.10e}, table {target:.10e}"
        )


def _forward_state():
    from tengri.protocols.component import ForwardState

    return ForwardState(wave=jnp.logspace(3.0, 4.5, 64))


def test_field_changes_the_sfr_shape(model, stellar):
    """The field still reshapes the SFR: at some ages the draw departs from the table by > 1 %."""
    rng = np.random.default_rng(5)
    out = evaluate_sfh(
        stellar.config,
        _params(model, rng.standard_normal(N_GRID)),
        jnp.asarray(SSP_AGES_YR),
        N_GRID,
    )
    lbt_yr = np.asarray(out.sfh_lbt_grid)
    table = _table_on(lbt_yr)
    inside = table > 0.0
    sfr = np.asarray(out.sfr_history)
    shape_sfr = sfr[inside] / sfr[inside].sum()
    shape_table = table[inside] / table[inside].sum()
    assert np.max(np.abs(shape_sfr / shape_table - 1.0)) > 1e-2


def _weights_at_xi_zero(ssp, sfh_model, n_grid, field, extra):
    """Joint weights and log formed mass at xi = 0, delta metallicity, z = Z_OBS."""
    from tengri.components.stellar.component import StellarSEDComponentConfig
    from tengri.protocols.component import ForwardState

    comp = StellarSEDComponent(
        config=StellarSEDComponentConfig(sfh_model=sfh_model, field=field, n_grid=n_grid),
        ssp_data=ssp,
    )
    params = {"redshift": jnp.asarray(Z_OBS), "met_logzsol": jnp.asarray(0.0), **extra}
    if field:
        params["sfh_field_psd_sigma"] = jnp.asarray(PSD_SIGMA)
        params["sfh_field_psd_tau_myr"] = jnp.asarray(PSD_TAU_MYR)
        params["sfh_field_xi"] = jnp.zeros(n_grid)
    state = comp.apply(ForwardState(wave=jnp.logspace(3.0, 4.5, 16)), params, ssp)
    return np.asarray(state.derived["joint_weights"]), float(state.derived["log_mstar_formed"])


@pytest.mark.parametrize("n_grid", [64, 512])
@pytest.mark.parametrize("kind", ["dpl", "table"])
def test_xi_zero_equals_no_field_to_grid_resolution(synthetic_ssp_wide, kind, n_grid):
    """At xi = 0 the field weights equal the no-field weights up to the log-grid interpolation.

    The field path interpolates the n_grid log-grid history to the SSP ages, the no-field
    path evaluates the SFR at the SSP ages: a parametric SFH on ``origin/main`` differs the
    same way. Measured max |dw| (z = 0.1): dpl 4.4e-3 (n_grid 64), 1.0e-4 (512); table 6.1e-3
    (64), 2.1e-3 (512), the same order as the stepped ``continuity`` history (5.4e-3, 2.6e-3).
    The formed mass is identical, and a smooth history's difference falls with n_grid.
    """
    extra = (
        {"sfh_t_gyr": jnp.asarray(T_NODES_GYR), "sfh_sfr": jnp.asarray(SFR_NODES)}
        if kind == "table"
        else {}
    )
    w_off, m_off = _weights_at_xi_zero(synthetic_ssp_wide, kind, n_grid, False, extra)
    w_on, m_on = _weights_at_xi_zero(synthetic_ssp_wide, kind, n_grid, True, extra)
    assert m_on == m_off
    diff = float(np.max(np.abs(w_on - w_off)))
    bound = {("dpl", 64): 1e-2, ("dpl", 512): 3e-4, ("table", 64): 1e-2, ("table", 512): 5e-3}
    assert diff <= bound[(kind, n_grid)]
