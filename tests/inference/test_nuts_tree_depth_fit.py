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


# --------------------------------------------------------------------------
# Warmup may be capped independently of sampling
# --------------------------------------------------------------------------
#
# The two halves are not symmetric. During warmup the step size has not
# converged, so trees are at their deepest exactly where they are least
# informative: `bench/reports/2026-08-31_fast_nuts.md` Finding 7 measured one
# vmapped 100-step adaptation over 64 lanes at D = 3 costing 1272 s, ~12.7 s per
# adaptation step for a three-parameter model, and Finding 2 measured warmup at
# 2.52x the cost of sampling on a single-galaxy fit. #2101 made
# `max_num_doublings` reach warmup at all; it reaches it with the *sampling*
# value, and nobody had asked whether warmup wants its own.
#
# `warmup_max_num_doublings=None` keeps the two equal, so the knob is a no-op
# unless asked for.


def test_warmup_cap_defaults_to_the_sampling_cap(synthetic_ssp, simple_observation):
    """Not passing the knob leaves the two halves equal — this must stay a no-op."""
    from tengri import ForwardModel

    model = _build_model(synthetic_ssp, simple_observation)
    _flux, noise, flux_obs = _mock(model)
    forward = ForwardModel.build(sed=model)
    post = forward.fit(
        flux_obs,
        noise,
        method="mcmc_nuts",
        key=jax.random.PRNGKey(2),
        n_warmup=40,
        n_burnin=5,
        n_samples=20,
        max_num_doublings=4,
        verbose=False,
    )
    d = post.diagnostics
    assert d["max_num_doublings"] == 4
    assert d["warmup_max_num_doublings"] == 4


def test_warmup_cap_reaches_the_adaptation_and_not_the_sampler(
    synthetic_ssp, simple_observation, monkeypatch
):
    """A shallower warmup cap must reach `window_adaptation` only.

    Behavioral rather than a source grep: it reads what the adaptation was
    actually constructed with, and separately what the fit reports for the
    sampling half.
    """
    import blackjax

    from tengri import ForwardModel

    model = _build_model(synthetic_ssp, simple_observation)
    _flux, noise, flux_obs = _mock(model)
    forward = ForwardModel.build(sed=model)

    seen: list[dict] = []
    real = blackjax.window_adaptation

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(blackjax, "window_adaptation", spy)
    post = forward.fit(
        flux_obs,
        noise,
        method="mcmc_nuts",
        key=jax.random.PRNGKey(2),
        n_warmup=40,
        n_burnin=5,
        n_samples=20,
        max_num_doublings=8,
        warmup_max_num_doublings=2,
        verbose=False,
    )

    assert seen, "the NUTS path must build a window adaptation"
    assert all(k.get("max_num_doublings") == 2 for k in seen), (
        f"warmup must adapt under the WARMUP cap, saw {seen}"
    )
    d = post.diagnostics
    assert d["warmup_max_num_doublings"] == 2
    assert d["max_num_doublings"] == 8, "the sampling half keeps its own cap"
    # And the sampler really did use the deeper cap, so the two are independent.
    assert d["tree_depth_max"] <= 8


def test_warmup_cap_is_in_the_adaptation_cache_key(synthetic_ssp, simple_observation):
    """A step size dual-averaged under one warmup cap is not another's.

    Two fits on one model at different warmup caps must not share an adaptation;
    if they did, the second would sample at the first's step size and nothing
    would say so.
    """
    from tengri import ForwardModel

    model = _build_model(synthetic_ssp, simple_observation)
    _flux, noise, flux_obs = _mock(model)
    forward = ForwardModel.build(sed=model)
    kw = dict(
        method="mcmc_nuts",
        key=jax.random.PRNGKey(2),
        n_warmup=40,
        n_burnin=5,
        n_samples=20,
        max_num_doublings=8,
        verbose=False,
    )
    shallow = forward.fit(flux_obs, noise, warmup_max_num_doublings=2, **kw)
    deep = forward.fit(flux_obs, noise, warmup_max_num_doublings=8, **kw)
    assert shallow.diagnostics["step_size"] != deep.diagnostics["step_size"], (
        "a shallower warmup cap must produce its own step size, not reuse the "
        "cached one from a different cap"
    )


def _mock(model):
    """Mock photometry for `model` at a fixed mass, deterministic in the seed."""
    key = jax.random.PRNGKey(0)
    truth = dict(model.spec.sample(key))
    truth["sfh_dpl_log_total_mass"] = jnp.array(10.0)
    flux = model.predict_photometry(truth)
    noise = jnp.abs(flux) * 0.02
    flux_obs = flux + noise * jax.random.normal(jax.random.fold_in(key, 1), shape=flux.shape)
    return flux, noise, flux_obs
