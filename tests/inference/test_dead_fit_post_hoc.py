# SPDX-License-Identifier: BSD-3-Clause
"""Post-hoc dead-fit detection after sampling completes (#2093).

The pre-hoc guard (#2088) refuses sampling when warmup ends with high divergence,
saving minutes of computation on dead fits. But a fit can survive warmup and still
return a frozen posterior: issue #2093 reproduced on nb05, seed 0, showing 100%
divergent draws and unique-row fraction 0.002 (1200 draws, 2 unique) — the model
had an unseen pathology the final-window warmup window statistic could not catch.

This file tests both the unit-level refuse_dead_sampling() function and the
end-to-end wiring that calls it from the NUTS backend.

Unit tests — Mutation checks:
1. ``test_all_divergent_raises``: drop the threshold check in refuse_dead_sampling
2. ``test_sub_threshold_quiet``: change ``<`` to ``<=`` in the threshold
3. ``test_at_threshold_raises``: same
4. ``test_none_record_quiet``: drop the None guard
5. ``test_empty_record_quiet``: drop the empty-record guard
6. ``test_two_step_hmc_record_is_not_judged``: remove the floor guard
   DEAD_SAMPLING_MIN_DRAWS (the R15 case: test_a_two_step_hmc_warmup_is_not_judged
   in test_dead_warmup_refusal_fit.py regresses if this guard is missing)
7. ``test_at_floor_raises``: change ``<`` to ``<=`` in the floor check

Wiring tests — Mutation checks:
1. ``test_nuts_raises_on_sampling_collapse_after_healthy_warmup``: remove the
   ``refuse_dead_sampling`` call in nuts.py.
2. ``test_healthy_fit_reports_sampling_diagnostics``: drop
   ``sampling_divergence_frac`` from the diagnostics dict, or ``unique_draw_frac``.
"""

import copy
import logging
import pickle
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import DeadFitError
from tengri.inference.backends.mcmc import nuts as nuts_backend
from tengri.inference.backends.mcmc._shared import (
    DEAD_SAMPLING_DIVERGENCE_FRAC,
    DEAD_WARMUP_DIVERGENCE_FRAC,
    refuse_dead_sampling,
    sampling_diagnostics,
)

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


# ── Unit tests on refuse_dead_sampling ────────────────────────────────


def test_constants_are_the_documented_values():
    assert DEAD_SAMPLING_DIVERGENCE_FRAC == 0.9


def test_all_divergent_raises():
    """100% divergent draws at threshold trigger the guard."""
    divergent = np.ones(100, dtype=bool)
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_sampling(
            divergent,
            sampler="NUTS",
            n_samples=100,
            n_chains=1,
            step_size=0.05,
        )
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0)
    assert err.step_size == pytest.approx(0.05)
    assert "sampling completed dead" in str(err)
    assert "100%" in str(err)
    assert "#2093" in str(err)


def test_sub_threshold_quiet():
    """89% divergent is below the 90% threshold; no raise."""
    divergent = np.concatenate([np.ones(89, dtype=bool), np.zeros(11, dtype=bool)])
    refuse_dead_sampling(
        divergent,
        sampler="NUTS",
        n_samples=100,
        n_chains=1,
        step_size=0.05,
    )


def test_at_threshold_raises():
    """90% divergent at exactly the threshold triggers the guard."""
    divergent = np.concatenate([np.ones(90, dtype=bool), np.zeros(10, dtype=bool)])
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_sampling(
            divergent,
            sampler="HMC",
            n_samples=100,
            n_chains=1,
            step_size=0.05,
        )
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(0.9)


def test_no_record_means_no_refusal():
    """None record (nothing measured) returns quietly."""
    refuse_dead_sampling(
        None,
        sampler="NUTS",
        n_samples=100,
        n_chains=1,
        step_size=0.05,
    )


def test_empty_record_measures_nothing():
    """Empty (0-length) divergent record returns quietly."""
    divergent = np.zeros(0, dtype=bool)
    refuse_dead_sampling(
        divergent,
        sampler="NUTS",
        n_samples=100,
        n_chains=1,
        step_size=0.05,
    )


def test_two_step_hmc_record_is_not_judged():
    """2/2 divergent draws return quietly (R15 case: dual-averaging opening burst).

    A record shorter than DEAD_SAMPLING_MIN_DRAWS is the dual-averaging opening
    burst and carries no verdict. The construction-time DeadFitWarning is the only
    guard entitled to judge it. Regresses test_a_two_step_hmc_warmup_is_not_judged
    in test_dead_warmup_refusal_fit.py if this floor is removed.
    """
    divergent = np.ones(2, dtype=bool)
    refuse_dead_sampling(
        divergent,
        sampler="HMC",
        n_samples=1,
        n_chains=2,
        step_size=0.05,
    )


def test_at_floor_raises():
    """10/10 divergent at exactly the floor (DEAD_SAMPLING_MIN_DRAWS) raises."""
    divergent = np.ones(10, dtype=bool)
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_sampling(
            divergent,
            sampler="NUTS",
            n_samples=10,
            n_chains=1,
            step_size=0.05,
        )
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0)


def test_multi_chain_flattened():
    """Multi-chain draws (n_chains=2, n_samples=50) flatten to (100,) divergent."""
    divergent = np.ones(100, dtype=bool)
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_sampling(
            divergent,
            sampler="Dynamic HMC",
            n_samples=50,
            n_chains=2,
            step_size=0.03,
        )
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0)
    assert "100" in str(err)


def test_one_dead_chain_raises_by_per_chain_max():
    """Per-chain check: one-dead-one-healthy at n_chains=2 raises on the dead chain.

    With n_chains=2, n_samples=50: first chain all divergent (50/50), second healthy
    (5/50). Aggregate mean is 55/100 = 55%, below the 90% threshold. The per-chain
    max is 100%, which raises. The error message should name chain 0 as dead.

    Mutation check: change the per-chain reshape or the `max()` to aggregate `mean()`
    and this test fails (one-dead-one-healthy silently passes).
    """
    # Chain 0: 50 divergent. Chain 1: 5 divergent out of 50.
    divergent = np.zeros(100, dtype=bool)
    divergent[:50] = True  # Chain 0: all divergent
    divergent[50:55] = True  # Chain 1: 5% divergent

    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_sampling(
            divergent,
            sampler="NUTS",
            n_samples=50,
            n_chains=2,
            step_size=0.05,
        )
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0), "Max across chains should be 100%"
    assert "chain 0" in str(err), "Error message should name the dead chain"
    # Aggregate would be 55/100=55%, but per-chain raises on max=100%.
    assert "sampling completed dead" in str(err)


def test_the_error_survives_pickling():
    """DeadFitError from post-hoc guard survives pickle/copy like pre-hoc."""
    divergent = np.ones(100, dtype=bool)
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_sampling(
            divergent,
            sampler="NUTS",
            n_samples=100,
            n_chains=1,
            step_size=0.0438,
        )
    err = excinfo.value
    for revived in (pickle.loads(pickle.dumps(err)), copy.copy(err)):
        assert isinstance(revived, DeadFitError)
        assert str(revived) == str(err)
        assert revived.sampling_divergence_frac == pytest.approx(1.0)
        assert revived.step_size == pytest.approx(0.0438)


def test_sampling_diagnostics_all_divergent():
    """sampling_diagnostics computes fractions from frozen fit."""
    positions = np.ones((100, 5)) * 0.5
    divergent = np.ones(100, dtype=bool)
    diag = sampling_diagnostics(positions, divergent)
    assert diag["sampling_divergence_frac"] == pytest.approx(1.0)
    assert diag["unique_draw_frac"] == pytest.approx(0.01)


def test_sampling_diagnostics_healthy_fit():
    """sampling_diagnostics on random draws shows low divergence and many unique."""
    np.random.seed(42)
    positions = np.random.randn(100, 5)
    divergent = np.zeros(100, dtype=bool)
    diag = sampling_diagnostics(positions, divergent)
    assert diag["sampling_divergence_frac"] == pytest.approx(0.0)
    assert diag["unique_draw_frac"] == pytest.approx(1.0)


def test_sampling_diagnostics_none_inputs():
    """None inputs to sampling_diagnostics return NaN (like pre-hoc)."""
    diag1 = sampling_diagnostics(None, np.ones(100, dtype=bool))
    assert np.isnan(diag1["sampling_divergence_frac"])
    assert np.isnan(diag1["unique_draw_frac"])

    diag2 = sampling_diagnostics(np.ones((100, 5)), None)
    assert np.isnan(diag2["sampling_divergence_frac"])
    assert np.isnan(diag2["unique_draw_frac"])


# ── Wiring tests on the backend integration ────────────────────────────


def test_nuts_raises_on_sampling_collapse_after_healthy_warmup(
    synthetic_ssp, simple_observation, monkeypatch
):
    """NUTS fit rejects all-divergent posteriors via post-hoc check (#2093).

    Warmup completes successfully, then the sampling scan returns all divergent
    draws. The fit must raise DeadFitError with sampling_divergence_frac == 1.0
    before Posterior assembly, and the message must name different-sampler
    remediation.

    This test exercises the seam where refuse_dead_sampling() is called in
    nuts.py, proving a mutant that removes the call would slip through the
    unit tests alone.
    """
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)

    # Monkeypatch both paths that can call the scan: the single-chain path
    # uses _nuts_chain_scan directly; the multi-chain path uses _vmap_chains.
    # Both must return all-divergent flags.
    original_chain_scan = nuts_backend._nuts_chain_scan
    original_vmap_chains = nuts_backend._vmap_chains
    chain_scan_called = []
    vmap_chains_called = []

    def patched_chain_scan(
        state, chain_keys, logdensity_fn_2arg, data_args, step_size, inv_mass_matrix, max_doublings
    ):
        chain_scan_called.append(True)
        positions, divergent_real, expansions = original_chain_scan(
            state,
            chain_keys,
            logdensity_fn_2arg,
            data_args,
            step_size,
            inv_mass_matrix,
            max_doublings,
        )
        divergent_patched = jnp.ones_like(divergent_real, dtype=bool)
        return positions, divergent_patched, expansions

    def patched_vmap_chains(_init, _scan, **kwargs):
        vmap_chains_called.append((kwargs["n_chains"], kwargs["n_iter"], kwargs["n_burnin"]))
        positions, divergent_real, expansions = original_vmap_chains(_init, _scan, **kwargs)
        # For multi-chain test: kill the first chain (indices 0 to n_samples-1 after
        # burnin is discarded). The reshape logic validates chain-major layout.
        n_samples = kwargs["n_iter"] - kwargs["n_burnin"]
        n_chains = kwargs["n_chains"]
        if n_chains == 2:
            # Make first chain 100% divergent, second healthy (10% divergent).
            divergent_patched = jnp.zeros_like(divergent_real, dtype=bool)
            # First chain all divergent
            divergent_patched = divergent_patched.at[:n_samples].set(True)
            # 10% of second chain divergent
            n_second_start = n_samples
            n_second_end = n_samples + n_samples // 10
            divergent_patched = divergent_patched.at[n_second_start:n_second_end].set(True)
        else:
            divergent_patched = jnp.ones_like(divergent_real, dtype=bool)
        return positions, divergent_patched, expansions

    monkeypatch.setattr(nuts_backend, "_nuts_chain_scan", patched_chain_scan)
    monkeypatch.setattr(nuts_backend, "_vmap_chains", patched_vmap_chains)

    # Test 1: Single-chain fit with all divergent should raise DeadFitError.
    with pytest.raises(DeadFitError) as excinfo:
        _fit(forward, flux_obs, noise, n_chains=1, verbose=False)
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0)
    assert "sampling completed dead" in str(err)
    assert "different sampler" in str(err).lower() or "sampler" in str(err)
    assert chain_scan_called, "Patched _nuts_chain_scan was never called"

    # Test 2: Two-chain fit where chain 0 is dead should raise and call vmap_chains.
    chain_scan_called.clear()
    vmap_chains_called.clear()
    with pytest.raises(DeadFitError) as excinfo:
        _fit(forward, flux_obs, noise, n_chains=2, verbose=False)
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0), "Dead chain should report 100%"
    assert "chain 0" in str(err), "Error message should name the dead chain"
    assert vmap_chains_called, "Patched _vmap_chains was never called for multi-chain path"


def test_healthy_fit_reports_sampling_diagnostics(synthetic_ssp, simple_observation, caplog):
    """A healthy unpatched fit reports sampling_divergence_frac and unique_draw_frac.

    Both diagnostics keys must be present in a successful fit. The sampling
    divergence fraction should be low (<0.5) and unique_draw_frac should be
    high (>0.8) on a healthy fit.
    """
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    caplog.set_level(logging.INFO, logger="tengri")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = _fit(forward, flux_obs, noise, verbose=True)

    # Diagnostics must be present.
    assert "sampling_divergence_frac" in post.diagnostics
    assert "unique_draw_frac" in post.diagnostics

    # Healthy fit should have low divergence and high unique fraction.
    sampling_div_frac = post.diagnostics["sampling_divergence_frac"]
    unique_frac = post.diagnostics["unique_draw_frac"]

    assert 0.0 <= sampling_div_frac < 0.5, f"Expected <0.5, got {sampling_div_frac:.1%}"
    assert 0.8 < unique_frac <= 1.0, f"Expected >0.8, got {unique_frac:.1%}"


def test_a_dead_sampling_evicts_the_cached_adaptation(
    synthetic_ssp, simple_observation, monkeypatch, caplog
):
    """Dead sampling refusal evicts the cached adaptation so next fit re-tunes.

    When post-hoc dead-fit detection raises, the cached adaptation (step size
    and mass matrix from the failed fit's warmup) must be evicted so the next
    fit with the same model re-runs warmup instead of reusing poisoned tuning.

    Mutation check: set the eviction condition to `if False and` in
    refuse_dead_sampling and this test fails (second fit silently reuses cache).
    """
    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    original_chain_scan = nuts_backend._nuts_chain_scan

    def patched_chain_scan(
        state, chain_keys, logdensity_fn_2arg, data_args, step_size, inv_mass_matrix, max_doublings
    ):
        positions, divergent_real, expansions = original_chain_scan(
            state,
            chain_keys,
            logdensity_fn_2arg,
            data_args,
            step_size,
            inv_mass_matrix,
            max_doublings,
        )
        divergent_patched = jnp.ones_like(divergent_real, dtype=bool)
        return positions, divergent_patched, expansions

    # Patch to force all divergent, run, expect DeadFitError
    monkeypatch.setattr(nuts_backend, "_nuts_chain_scan", patched_chain_scan)
    with pytest.raises(DeadFitError):
        _fit(forward, flux_obs, noise, n_chains=1, verbose=False)

    # Undo patch and run the same fit again unpatched
    monkeypatch.undo()
    caplog.set_level(logging.INFO, logger="tengri")
    caplog.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = _fit(forward, flux_obs, noise, n_chains=1, verbose=True)

    # Assert warmup re-ran: diagnostics contain warmup_divergence_frac.
    # A cached fit omits this key entirely (see test_a_reused_adaptation_carries_no_warmup_record)
    assert "warmup_divergence_frac" in post.diagnostics
    assert 0.0 <= post.diagnostics["warmup_divergence_frac"] < DEAD_WARMUP_DIVERGENCE_FRAC, (
        "Re-run warmup should be healthy"
    )

    # Verify no cache-reuse log message appeared
    messages = [r.getMessage() for r in caplog.records]
    assert not any("Reusing cached warmup" in m for m in messages), (
        "Second fit should not reuse cache after eviction"
    )


def test_hmc_raises_on_sampling_collapse_after_healthy_warmup(
    synthetic_ssp, simple_observation, monkeypatch
):
    """HMC wiring: refusal when sampling diverges despite healthy warmup.

    Patched _hmc_chain_scan forces all divergent flags. HMC's refuse_dead_sampling
    call should catch this and raise DeadFitError. Mutation check: remove the
    refuse_dead_sampling call in hmc.py and this test fails.
    """
    from tengri.inference.backends.mcmc import hmc as hmc_backend

    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    original_chain_scan = hmc_backend._hmc_chain_scan

    def patched_hmc_chain_scan(*args, **kwargs):
        positions, divergent_real = original_chain_scan(*args, **kwargs)
        divergent_patched = jnp.ones_like(divergent_real, dtype=bool)
        return positions, divergent_patched

    monkeypatch.setattr(hmc_backend, "_hmc_chain_scan", patched_hmc_chain_scan)

    # Single-chain fit should raise DeadFitError.
    with pytest.raises(DeadFitError) as excinfo:
        _fit(forward, flux_obs, noise, method="mcmc_hmc", n_chains=1, verbose=False)
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0)
    assert "sampling completed dead" in str(err)


def test_dynamic_hmc_raises_on_sampling_collapse_after_healthy_warmup(
    synthetic_ssp, simple_observation, monkeypatch
):
    """Dynamic HMC wiring: refusal when sampling diverges despite healthy warmup.

    Patched _dynamic_hmc_chain_scan forces all divergent flags. Dynamic HMC's
    refuse_dead_sampling call should catch this and raise DeadFitError. Mutation
    check: remove the refuse_dead_sampling call in dynamic_hmc.py and this test fails.
    """
    from tengri.inference.backends.mcmc import dynamic_hmc as dhmc_backend

    forward, flux_obs, noise = _forward_and_data(synthetic_ssp, simple_observation)
    original_chain_scan = dhmc_backend._dynamic_hmc_chain_scan

    def patched_dhmc_chain_scan(*args, **kwargs):
        positions, divergent_real = original_chain_scan(*args, **kwargs)
        divergent_patched = jnp.ones_like(divergent_real, dtype=bool)
        return positions, divergent_patched

    monkeypatch.setattr(dhmc_backend, "_dynamic_hmc_chain_scan", patched_dhmc_chain_scan)

    # Single-chain fit should raise DeadFitError.
    with pytest.raises(DeadFitError) as excinfo:
        _fit(forward, flux_obs, noise, method="mcmc_dynamic_hmc", n_chains=1, verbose=False)
    err = excinfo.value
    assert err.sampling_divergence_frac == pytest.approx(1.0)
    assert "sampling completed dead" in str(err)
