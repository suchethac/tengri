# SPDX-License-Identifier: BSD-3-Clause
"""The dead-warmup refusal at the backend seam, and the healthy-path plumbing (#2088).

One real two-chain NUTS fit on the synthetic SSP checks that the divergence
record reaches ``diagnostics`` and that the log lines count every chain
(#2087). The refusal itself is exercised by replacing the warmup helper with
one that returns an all-divergent record: the fit must raise before the
sampling scan is entered and must not cache the adaptation.

Mutation checks:
1. ``test_nuts_refuses_a_dead_warmup_before_sampling``: remove the
   ``refuse_dead_warmup`` call in nuts.py.
2. ``test_a_refused_warmup_is_not_cached``: move ``_set_cached_adaptation``
   above the refusal.
3. ``test_healthy_two_chain_fit_reports_the_record_and_counts_every_chain``:
   drop ``warmup_divergence_frac`` from the diagnostics dict, or revert the
   log line to ``n_samples``.
4. ``test_a_reused_adaptation_carries_no_warmup_record``: put
   ``"warmup_divergence_frac": None`` back on the cached branch of nuts.py.
5. ``test_hmc_refuses_a_dead_warmup_before_sampling``: remove the
   ``refuse_dead_warmup`` call in hmc.py / dynamic_hmc.py.
6. ``test_a_two_step_hmc_warmup_is_not_judged``: remove the
   ``flags.size < DEAD_WARMUP_MIN_WINDOW`` guard in
   ``final_window_divergence_frac`` -- the fit then raises.
"""

import logging
import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri.config.exceptions import DeadFitError, DeadFitWarning
from tengri.inference.backends.mcmc import (
    dynamic_hmc as dynamic_hmc_backend,
    hmc as hmc_backend,
    nuts as nuts_backend,
)
from tengri.inference.backends.mcmc._shared import DEAD_WARMUP_DIVERGENCE_FRAC

# The synthetic SSP bakes its nebular emission in, and both ``SEDModel(...)``
# and every ``fit`` re-announce it. That notice is the fixture's, not this
# file's subject; the DeadFitWarning assertion below records its own warnings.
pytestmark = [
    pytest.mark.contract,
    pytest.mark.filterwarnings("ignore::tengri.components.nebular.BakedInNebularWarning"),
]

_MASS_LO, _MASS_HI = 7.0, 12.5
_N_WARMUP, _N_BURNIN, _N_SAMPLES, _N_CHAINS = 40, 5, 20, 2


def _forward_and_data(synthetic_ssp, simple_observation):
    """One-free-parameter dpl model: mass only, everything else pinned."""
    from tengri import Fixed, ForwardModel, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(_MASS_LO, _MASS_HI),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    model = SEDModel(spec, synthetic_ssp, observation=simple_observation)
    key = jax.random.PRNGKey(0)
    truth = dict(model.spec.sample(key))
    truth["sfh_dpl_log_total_mass"] = jnp.array(10.0)
    flux = model.predict_photometry(truth)
    noise = jnp.abs(flux) * 0.02
    flux_obs = flux + noise * jax.random.normal(jax.random.fold_in(key, 1), shape=flux.shape)
    return ForwardModel.build(sed=model), flux_obs, noise


def _fit(forward, flux_obs, noise, **overrides):
    kwargs = dict(
        method="mcmc_nuts",
        key=jax.random.PRNGKey(2),
        n_warmup=_N_WARMUP,
        n_burnin=_N_BURNIN,
        n_samples=_N_SAMPLES,
        n_chains=_N_CHAINS,
        verbose=True,
    )
    kwargs.update(overrides)
    return forward.fit(flux_obs, noise, **kwargs)


def _dead_warmup(*args, **kwargs):
    n_warmup = args[4]
    return jnp.array(0.01), jnp.ones((1,)), jnp.ones((n_warmup,), dtype=bool)


def test_nuts_refuses_a_dead_warmup_before_sampling(
    synthetic_ssp, simple_observation, monkeypatch
):
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    monkeypatch.setattr(nuts_backend, "_nuts_warmup_only", _dead_warmup)

    def _sampling_must_not_run(*args, **kwargs):
        raise AssertionError("the sampling scan ran after a dead warmup")

    monkeypatch.setattr(nuts_backend, "_nuts_chain_scan", _sampling_must_not_run)
    with pytest.raises(DeadFitError) as excinfo:
        _fit(forward, flux_obs, noise)
    assert excinfo.value.warmup_divergence_frac == pytest.approx(1.0)
    assert "NUTS" in str(excinfo.value)


def test_a_refused_warmup_is_not_cached(synthetic_ssp, simple_observation, monkeypatch, caplog):
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    monkeypatch.setattr(nuts_backend, "_nuts_warmup_only", _dead_warmup)
    with pytest.raises(DeadFitError):
        _fit(forward, flux_obs, noise)
    monkeypatch.undo()

    caplog.set_level(logging.INFO, logger="tengri")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit(forward, flux_obs, noise)
    messages = [r.getMessage() for r in caplog.records]
    assert not any("Reusing cached warmup" in m for m in messages), messages
    assert any("Warmup complete" in m for m in messages), messages


def test_healthy_two_chain_fit_reports_the_record_and_counts_every_chain(
    synthetic_ssp, simple_observation, caplog
):
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    caplog.set_level(logging.INFO, logger="tengri")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        post = _fit(forward, flux_obs, noise)
    assert not [w for w in record if issubclass(w.category, DeadFitWarning)]

    diag = post.diagnostics
    assert diag["n_chains"] == _N_CHAINS
    assert 0.0 <= diag["warmup_divergence_frac"] < DEAD_WARMUP_DIVERGENCE_FRAC
    assert len(next(iter(post.samples.values()))) == _N_CHAINS * _N_SAMPLES

    completion = [r.getMessage() for r in caplog.records if "NUTS complete" in r.getMessage()]
    assert len(completion) == 1, caplog.records
    assert f"/{_N_CHAINS * _N_SAMPLES} (" in completion[0]
    assert "Tree depth: mean" in completion[0]
    warmup_lines = [r.getMessage() for r in caplog.records if "Warmup complete" in r.getMessage()]
    assert warmup_lines and "final warmup window" in warmup_lines[0]


def test_a_reused_adaptation_carries_no_warmup_record(synthetic_ssp, simple_observation, caplog):
    """A warm fit measured no warmup, so it reports no fraction -- not ``None``.

    ``Posterior.save()`` has no HDF5 representation for ``None`` and warns
    about every entry it has to skip, so a key that is only sometimes
    meaningful must be absent rather than null on the other branch (#2088).
    """
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    caplog.set_level(logging.INFO, logger="tengri")
    first = _fit(forward, flux_obs, noise)
    caplog.clear()
    second = _fit(forward, flux_obs, noise)

    # The second call must actually have taken the cached-adaptation branch,
    # or the assertion below would pass for the wrong reason.
    assert any("Reusing cached warmup" in r.getMessage() for r in caplog.records), caplog.records

    assert "warmup_divergence_frac" in first.diagnostics
    assert "warmup_divergence_frac" not in second.diagnostics


# Both HMC backends import ``_hmc_warmup_only`` into their own namespace and
# call it positionally, so ``args[4]`` is ``n_warmup`` in each and one stub
# serves both seams.
@pytest.mark.parametrize(
    ("method", "backend", "sampler"),
    [
        ("mcmc_hmc", hmc_backend, "HMC"),
        ("mcmc_dynamic_hmc", dynamic_hmc_backend, "Dynamic HMC"),
    ],
)
def test_hmc_refuses_a_dead_warmup_before_sampling(
    synthetic_ssp, simple_observation, monkeypatch, method, backend, sampler
):
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    monkeypatch.setattr(backend, "_hmc_warmup_only", _dead_warmup)
    with pytest.raises(DeadFitError) as excinfo:
        _fit(forward, flux_obs, noise, method=method)
    assert excinfo.value.warmup_divergence_frac == pytest.approx(1.0)
    assert sampler in str(excinfo.value)


def test_a_two_step_hmc_warmup_is_not_judged(synthetic_ssp, simple_observation):
    """A warmup too short to fill the minimum window carries no verdict (#2088, R15).

    BlackJAX opens dual averaging at ``mu = log(10 * step_size)``, so the first
    proposals are made at roughly twice the initial step size whatever the
    posterior and take five or six rejections to collapse. A two-step record is
    that opening burst and nothing else, on a healthy posterior as much as on a
    dead one, so it must not refuse the fit -- and must not report a fraction
    either.

    Both kept draws do diverge at that un-collapsed step size, which is exactly
    the point: the construction-time guard still says so, and it is the only
    guard entitled to, because it judges the draws rather than the opening of
    an adaptation that never got to finish.
    """
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    with pytest.warns(DeadFitWarning, match=r"2/2 divergent"):
        post = _fit(
            forward,
            flux_obs,
            noise,
            method="mcmc_hmc",
            n_warmup=2,
            n_burnin=0,
            n_samples=2,
            n_chains=1,
        )
    assert post.samples
    assert "warmup_divergence_frac" not in post.diagnostics
