# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end check that a NUTS fit reports tree-depth diagnostics.

The unit tier pins the stats helper and the warning gate in isolation;
this test runs the real sampler on the synthetic SSP and asserts the
per-iteration trajectory-expansion counts actually reach the Posterior's
diagnostics dict — the plumbing a unit test cannot see (scan payload
through ``_nuts_chain_scan`` and the burnin slice in ``run_nuts``).
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

_MASS_LO, _MASS_HI = 7.0, 12.5


def _build_model(synthetic_ssp, simple_observation):
    """One-free-parameter dpl model: mass only, everything else pinned."""
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(_MASS_LO, _MASS_HI),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),  # in-grid for synthetic_ssp
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


def test_nuts_diagnostics_carry_tree_depth(synthetic_ssp, simple_observation):
    from tengri import ForwardModel

    model = _build_model(synthetic_ssp, simple_observation)
    key = jax.random.PRNGKey(0)
    truth = dict(model.spec.sample(key))
    truth["sfh_dpl_log_total_mass"] = jnp.array(10.0)
    flux = model.predict_photometry(truth)
    noise = jnp.abs(flux) * 0.02
    flux_obs = flux + noise * jax.random.normal(jax.random.fold_in(key, 1), shape=flux.shape)

    forward = ForwardModel.build(sed=model)
    post = forward.fit(
        flux_obs,
        noise,
        method="mcmc_nuts",
        key=jax.random.PRNGKey(2),
        n_warmup=50,
        n_burnin=10,
        n_samples=30,
        verbose=False,
    )

    d = post.diagnostics
    assert d["max_num_doublings"] >= 1
    assert 1 <= d["tree_depth_max"] <= d["max_num_doublings"]
    assert 0.0 < d["tree_depth_mean"] <= d["tree_depth_max"]
    assert 0.0 <= d["frac_max_depth"] <= 1.0


def test_multichain_nuts_also_reports_depth(synthetic_ssp, simple_observation):
    """The vmapped multi-chain branch flattens a different payload shape."""
    from tengri import ForwardModel

    model = _build_model(synthetic_ssp, simple_observation)
    key = jax.random.PRNGKey(3)
    truth = dict(model.spec.sample(key))
    truth["sfh_dpl_log_total_mass"] = jnp.array(10.0)
    flux = model.predict_photometry(truth)
    noise = jnp.abs(flux) * 0.02

    forward = ForwardModel.build(sed=model)
    post = forward.fit(
        flux + noise,
        noise,
        method="mcmc_nuts",
        key=jax.random.PRNGKey(4),
        n_warmup=50,
        n_burnin=10,
        n_samples=30,
        n_chains=2,
        verbose=False,
    )
    d = post.diagnostics
    assert 1 <= d["tree_depth_max"] <= d["max_num_doublings"]
    assert 0.0 <= d["frac_max_depth"] <= 1.0
