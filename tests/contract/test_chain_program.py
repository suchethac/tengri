# SPDX-License-Identifier: BSD-3-Clause
"""The adapted-chain runner reproduces the NUTS scans bit for bit.

``chain_program`` replaces the per-family scans in ``backends/mcmc/_shared.py``
with one adaptation and one sampling body. Each entry point here is compared with
a raw BlackJAX oracle on the same keys and the same statics, so a change in key
handling, scan carry or collected outputs shows up as a failed equality rather
than as a shifted posterior. The fused ``run_chain`` call is compared with
``run_chain`` directly. The compiled-program checks pin the shape of the lowered
control flow and the number of compiled entries.

The target is the anisotropic Gaussian of ``test_branch_free_samplers.py``.
"""

from __future__ import annotations

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import BackendError
from tengri.inference._backend_registry import all_backends, get_backend
from tengri.inference.backends.mcmc import _shared
from tengri.inference.backends.mcmc.chain_program import (
    DUAL_AVERAGING,
    DYNAMIC_HMC,
    DYNAMIC_HMC_TUNING_LEAPFROG_STEPS,
    FIRST_ORDER,
    GHMC,
    HMC,
    MEADS,
    NUTS,
    PATHFINDER_NUTS,
    WINDOW_GHMC,
    WINDOW_HMC,
    WINDOW_NUTS,
    FirstOrderConfig,
    GHMCConfig,
    HMCConfig,
    NUTSConfig,
    adapt_chain,
    divergence_record,
    run_chain,
    sample_chain,
)
from tests.contract.test_inference_cache_keys import fitter_photometry

pytestmark = pytest.mark.contract

N_DIM = 4
N_WARMUP = 30
N_ITER = 30
CFG = NUTSConfig(max_doublings=10, use_dense=False, target_accept_rate=0.8)


def _target(position, data_args):
    """Anisotropic Gaussian log-density, in the ``(position, data_args)`` shape."""
    return -0.5 * jnp.sum((position / data_args) ** 2)


def _raw_sampling_scan(state, chain_keys, scales, step_size, inv_mass):
    """Raw BlackJAX NUTS scan with fixed tuning: the oracle for ``sample_chain``."""
    kernel = blackjax.nuts.build_kernel()

    def log_density(position):
        return _target(position, scales)

    def _step(s, k):
        s, info = kernel(k, s, log_density, step_size, inv_mass, CFG.max_doublings)
        return s, (
            s.position,
            info.is_divergent,
            info.num_trajectory_expansions,
            info.num_integration_steps,
            info.energy,
        )

    _, outputs = jax.lax.scan(_step, state, chain_keys)
    return outputs


def _raw_warmup(init, warmup_key, scales, n_warmup, pathfinder):
    """Raw BlackJAX adaptation: the oracle for ``adapt_chain`` and ``run_chain``.

    Returns the adapted state, the step size, inverse mass matrix, per-step
    divergence flags and the leapfrog count summed over the adaptation.
    """

    def log_density(position):
        return _target(position, scales)

    if pathfinder:
        from blackjax.adaptation.pathfinder_adaptation import pathfinder_adaptation

        warmup = pathfinder_adaptation(
            blackjax.nuts,
            log_density,
            target_acceptance_rate=CFG.target_accept_rate,
            max_num_doublings=CFG.max_doublings,
        )
        with _shared._bounded_pathfinder_elbo_draws():
            (state, params), info = warmup.run(warmup_key, init, num_steps=n_warmup)
    else:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_density,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=CFG.target_accept_rate,
            max_num_doublings=CFG.max_doublings,
        )
        (state, params), info = warmup.run(warmup_key, init, num_steps=n_warmup)
    return (
        state,
        params["step_size"],
        params["inverse_mass_matrix"],
        info.info.is_divergent,
        jnp.sum(info.info.num_integration_steps),
    )


@pytest.fixture(scope="module")
def setup():
    """Start position, warmup key, per-iteration keys and per-coordinate scales."""
    scales = jnp.asarray([10.0**i for i in range(N_DIM)])
    return (
        jnp.zeros(N_DIM),
        jax.random.PRNGKey(0),
        jax.random.split(jax.random.PRNGKey(1), N_ITER),
        scales,
    )


def test_sample_chain_matches_raw_nuts_scan(setup):
    """Sampling with fixed adaptation equals the raw BlackJAX scan, output for output."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.5)
    inv_mass = scales**2
    state = blackjax.mcmc.nuts.init(init, lambda p: _target(p, scales))

    expected = _raw_sampling_scan(state, keys, scales, step_size, inv_mass)
    got = sample_chain(NUTS, state, keys, _target, scales, (step_size, inv_mass), CFG.sampling())
    actual = (
        got.position,
        got.divergent,
        got.extras["expansions"],
        got.extras["n_leapfrog"],
        got.extras["energy"],
    )
    assert len(actual) == len(expected)
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_adapt_chain_matches_raw_window_adaptation(setup):
    """Window adaptation equals the raw BlackJAX adaptation: step size, mass, flags and cost."""
    init, wkey, _, scales = setup
    params, warmup_divergent, warmup_n_grad = adapt_chain(
        NUTS, WINDOW_NUTS, init, wkey, _target, scales, N_WARMUP, CFG
    )
    expected = _raw_warmup(init, wkey, scales, N_WARMUP, pathfinder=False)[1:]
    step_size, inv_mass = params
    for want, have in zip(
        expected, (step_size, inv_mass, warmup_divergent, warmup_n_grad), strict=True
    ):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_pathfinder_adaptation_matches_raw_pathfinder(setup):
    """Pathfinder adaptation equals the raw BlackJAX Pathfinder adaptation."""
    init, wkey, _, scales = setup
    params, warmup_divergent, warmup_n_grad = adapt_chain(
        NUTS, PATHFINDER_NUTS, init, wkey, _target, scales, N_WARMUP, CFG
    )
    expected = _raw_warmup(init, wkey, scales, N_WARMUP, pathfinder=True)[1:]
    step_size, inv_mass = params
    for want, have in zip(
        expected, (step_size, inv_mass, warmup_divergent, warmup_n_grad), strict=True
    ):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_run_chain_matches_raw_warmup_then_sampling(setup):
    """The fused program equals the raw adaptation followed by the raw scan on the same keys.

    The raw adaptation's final state is the sampling scan's start, so a wrong state
    handoff inside ``run_chain`` shows up as a failed equality.
    """
    init, wkey, keys, scales = setup
    draws, params = run_chain(
        NUTS, WINDOW_NUTS, init, wkey, None, keys, _target, scales, N_WARMUP, CFG
    )
    state, step_size, inv_mass, _, _ = _raw_warmup(init, wkey, scales, N_WARMUP, pathfinder=False)
    expected = _raw_sampling_scan(state, keys, scales, step_size, inv_mass)
    actual = (
        draws.position,
        draws.divergent,
        draws.extras["expansions"],
        draws.extras["n_leapfrog"],
        draws.extras["energy"],
    )
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))
    for want, have in zip((step_size, inv_mass), params, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_sampling_scan_ignores_the_adaptation_only_statics(setup):
    """Two sampling configs that differ only in target acceptance share one compiled scan."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.5)
    inv_mass = scales**2
    state = blackjax.mcmc.nuts.init(init, lambda p: _target(p, scales))
    tuned = NUTSConfig(max_doublings=10, use_dense=False, target_accept_rate=0.8)
    retargeted = NUTSConfig(max_doublings=10, use_dense=False, target_accept_rate=0.9)

    sample_chain(NUTS, state, keys, _target, scales, (step_size, inv_mass), tuned.sampling())
    size_before = sample_chain._cache_size()
    sample_chain(NUTS, state, keys, _target, scales, (step_size, inv_mass), retargeted.sampling())
    assert sample_chain._cache_size() == size_before


def test_sample_and_run_lower_to_the_same_control_flow(setup):
    """The sampling entry lowers to the same while ops and lines as the raw scan."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.5)
    inv_mass = scales**2
    state = blackjax.mcmc.nuts.init(init, lambda p: _target(p, scales))

    ours = sample_chain.lower(
        NUTS, state, keys, _target, scales, (step_size, inv_mass), CFG.sampling()
    )
    theirs = jax.jit(_raw_sampling_scan).lower(state, keys, scales, step_size, inv_mass)
    ours_text = ours.as_text()
    theirs_text = theirs.as_text()
    assert ours_text.count("stablehlo.while") == theirs_text.count("stablehlo.while")
    assert len(ours_text.splitlines()) == len(theirs_text.splitlines())


def test_run_chain_does_not_recompile_for_new_keys(setup):
    """Repeated calls with the same statics and new keys reuse one compiled entry."""
    init, wkey, keys, scales = setup
    run_chain(NUTS, WINDOW_NUTS, init, wkey, None, keys, _target, scales, N_WARMUP, CFG)
    size_after_first = run_chain._cache_size()
    for seed in (2, 3):
        new_keys = jax.random.split(jax.random.PRNGKey(seed), N_ITER)
        run_chain(NUTS, WINDOW_NUTS, init, wkey, None, new_keys, _target, scales, N_WARMUP, CFG)
    assert run_chain._cache_size() == size_after_first


def test_divergence_record_reduces_the_mask_once():
    """No flags give no record; flags give the count and the mask."""
    assert divergence_record(None) == {}
    mask = jnp.asarray([False, True, False, True])
    record = divergence_record(mask)
    assert int(record["n_divergent"]) == 2
    assert np.array_equal(np.asarray(record["divergent_mask"]), np.asarray(mask))


HMC_LEAPFROG = 10
HMC_CFG = HMCConfig(n_leapfrog=HMC_LEAPFROG, use_dense=False, target_accept_rate=0.85)


def _raw_hmc_sampling_scan(state, chain_keys, scales, step_size, inv_mass):
    """Raw BlackJAX HMC scan with fixed tuning: the oracle for ``sample_chain(HMC)``."""
    kernel = blackjax.mcmc.hmc.build_kernel()

    def log_density(position):
        return _target(position, scales)

    def _step(s, k):
        s, info = kernel(k, s, log_density, step_size, inv_mass, HMC_LEAPFROG)
        return s, (s.position, info.is_divergent)

    _, outputs = jax.lax.scan(_step, state, chain_keys)
    return outputs


def _raw_hmc_warmup(init, warmup_key, scales, n_warmup):
    """Raw BlackJAX HMC window adaptation: the oracle for ``adapt_chain(HMC, ...)``."""

    def log_density(position):
        return _target(position, scales)

    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        log_density,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=HMC_CFG.target_accept_rate,
        num_integration_steps=HMC_LEAPFROG,
    )
    (state, params), info = warmup.run(warmup_key, init, num_steps=n_warmup)
    return (
        state,
        params["step_size"],
        params["inverse_mass_matrix"],
        info.info.is_divergent,
    )


def test_hmc_sample_chain_matches_raw_hmc_scan(setup):
    """Sampling HMC with fixed adaptation equals the raw BlackJAX scan, output for output."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.5)
    inv_mass = scales**2
    state = blackjax.mcmc.hmc.init(init, lambda p: _target(p, scales))

    expected = _raw_hmc_sampling_scan(state, keys, scales, step_size, inv_mass)
    got = sample_chain(HMC, state, keys, _target, scales, (step_size, inv_mass), HMC_CFG)
    actual = (got.position, got.divergent)
    assert len(actual) == len(expected)
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_hmc_adapt_chain_matches_raw_window_adaptation(setup):
    """HMC window adaptation equals the raw BlackJAX adaptation: step, mass and flags."""
    init, wkey, _keys, scales = setup
    _, step_size, inv_mass, divergent = _raw_hmc_warmup(init, wkey, scales, N_WARMUP)
    (got_step, got_mass), got_divergent, _ = adapt_chain(
        HMC, WINDOW_HMC(HMC_LEAPFROG, True), init, wkey, _target, scales, N_WARMUP, HMC_CFG
    )
    assert np.array_equal(np.asarray(got_step), np.asarray(step_size))
    assert np.array_equal(np.asarray(got_mass), np.asarray(inv_mass))
    assert np.array_equal(np.asarray(got_divergent), np.asarray(divergent))


def test_hmc_run_chain_matches_raw_warmup_then_sampling(setup):
    """Fused HMC equals the raw adaptation followed by the raw scan, on the same keys."""
    init, wkey, keys, scales = setup
    state, step_size, inv_mass, _ = _raw_hmc_warmup(init, wkey, scales, N_WARMUP)
    expected = _raw_hmc_sampling_scan(state, keys, scales, step_size, inv_mass)

    draws, (got_step, got_mass) = run_chain(
        HMC,
        WINDOW_HMC(HMC_LEAPFROG, True),
        init,
        wkey,
        None,
        keys,
        _target,
        scales,
        N_WARMUP,
        HMC_CFG,
    )
    assert np.array_equal(np.asarray(got_step), np.asarray(step_size))
    assert np.array_equal(np.asarray(got_mass), np.asarray(inv_mass))
    actual = (draws.position, draws.divergent)
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_hmc_adaptation_rule_refuses_an_unresolved_dense_policy():
    """The registered HMC rule takes a decided mass policy, never the auto ``None``."""
    from tengri.inference._backend_registry import get_backend

    entry = get_backend("mcmc_hmc")
    with pytest.raises(ValueError, match="resolve_dense_mass_gate"):
        entry.adaptation_for({"dense_mass_matrix": None})
    dense = entry.adaptation_for({"dense_mass_matrix": True, "n_leapfrog_steps": 7})
    diagonal = entry.adaptation_for({"dense_mass_matrix": False, "n_leapfrog_steps": 7})
    assert dense.name != diagonal.name


def _raw_dynamic_hmc_scan(state, chain_keys, scales, step_size, inv_mass):
    """Raw BlackJAX dynamic HMC scan with fixed tuning: the oracle for ``sample_chain``."""
    kernel = blackjax.mcmc.dynamic_hmc.build_kernel()

    def log_density(position):
        return _target(position, scales)

    def _step(s, k):
        s, info = kernel(k, s, log_density, step_size, inv_mass)
        return s, (s.position, info.is_divergent)

    _, outputs = jax.lax.scan(_step, state, chain_keys)
    return outputs


def _raw_dynamic_hmc_state(init, init_key, scales):
    """Raw BlackJAX dynamic HMC state at ``init``, built from ``init_key``."""

    def log_density(position):
        return _target(position, scales)

    return blackjax.mcmc.dynamic_hmc.init(init, log_density, init_key)


def test_dynamic_hmc_sample_chain_matches_raw_scan(setup):
    """Sampling dynamic HMC from a fresh state equals the raw BlackJAX scan, output for output."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.5)
    inv_mass = scales**2
    state = _raw_dynamic_hmc_state(init, jax.random.PRNGKey(7), scales)

    expected = _raw_dynamic_hmc_scan(state, keys, scales, step_size, inv_mass)
    got = sample_chain(DYNAMIC_HMC, state, keys, _target, scales, (step_size, inv_mass), None)
    actual = (got.position, got.divergent)
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_dynamic_hmc_run_chain_matches_raw_warmup_then_sampling(setup):
    """Fused dynamic HMC equals HMC warmup, then a raw dynamic state built from the init key."""
    init, wkey, keys, scales = setup
    init_key = jax.random.PRNGKey(7)
    _, step_size, inv_mass, _ = _raw_hmc_warmup(init, wkey, scales, N_WARMUP)
    state = _raw_dynamic_hmc_state(init, init_key, scales)
    expected = _raw_dynamic_hmc_scan(state, keys, scales, step_size, inv_mass)

    draws, (got_step, got_mass) = run_chain(
        DYNAMIC_HMC,
        WINDOW_HMC(HMC_LEAPFROG, True),
        init,
        wkey,
        init_key,
        keys,
        _target,
        scales,
        N_WARMUP,
        HMC_CFG,
    )
    assert np.array_equal(np.asarray(got_step), np.asarray(step_size))
    assert np.array_equal(np.asarray(got_mass), np.asarray(inv_mass))
    actual = (draws.position, draws.divergent)
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_dynamic_hmc_adaptation_rule_refuses_an_unresolved_dense_policy():
    """The registered dynamic HMC rule takes a decided mass policy, never the auto ``None``."""
    from tengri.inference._backend_registry import get_backend

    entry = get_backend("mcmc_dynamic_hmc")
    assert entry.chain_program is DYNAMIC_HMC
    with pytest.raises(ValueError, match="resolve_dense_mass_gate"):
        entry.adaptation_for({"dense_mass_matrix": None})
    dense = entry.adaptation_for({"dense_mass_matrix": True})
    diagonal = entry.adaptation_for({"dense_mass_matrix": False})
    assert dense.name != diagonal.name


GHMC_ALPHA = 0.8
GHMC_DELTA = 0.65
GHMC_CFG = GHMCConfig(alpha=GHMC_ALPHA, delta=GHMC_DELTA, target_accept_rate=0.85)
MEADS_CFG = GHMCConfig(alpha=None, delta=None, n_ensemble=16, n_folds=4, jitter_scale=0.5)
MEADS_WARMUP = 100
MEADS_CHAINS = 2
MEADS_ITER = 100
#: Mildly anisotropic target: at the setup fixture's 10^i scales MEADS' step collapses to
#: ~0.009, every proposal diverges and no chain moves, so it cannot test the sampler.
MEADS_SCALES = jnp.asarray([1.0, 1.5, 2.0, 2.5])


def _raw_ghmc_sampling_scan(state, chain_keys, scales, step_size, inv_mass):
    """Raw BlackJAX GHMC scan with static alpha and delta: the oracle for sample_chain."""
    kernel = blackjax.mcmc.ghmc.build_kernel()

    def log_density(position):
        return _target(position, scales)

    def _step(s, k):
        s, info = kernel(k, s, log_density, step_size, inv_mass, GHMC_ALPHA, GHMC_DELTA)
        return s, (s.position, info.is_divergent)

    _, outputs = jax.lax.scan(_step, state, chain_keys)
    return outputs


def _raw_ghmc_init(init, init_key, scales):
    """Raw BlackJAX GHMC state at ``init``: the oracle for ``GHMC.init``."""

    def log_density(position):
        return _target(position, scales)

    return blackjax.mcmc.ghmc.init(position=init, logdensity_fn=log_density, rng_key=init_key)


def test_ghmc_sample_chain_matches_raw_ghmc_scan(setup):
    """Sampling GHMC with fixed tuning equals the raw BlackJAX scan, output for output."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.2)
    inv_mass = 1.0 / scales**2
    state = _raw_ghmc_init(init, jax.random.PRNGKey(7), scales)

    expected = _raw_ghmc_sampling_scan(state, keys, scales, step_size, inv_mass)
    assert np.any(np.asarray(expected[0]) != np.asarray(init)), "the chain must move"
    got = sample_chain(GHMC, state, keys, _target, scales, (step_size, inv_mass), GHMC_CFG)
    actual = (got.position, got.divergent)
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_ghmc_run_chain_window_matches_raw_warmup_then_sampling(setup):
    """Fused GHMC with window tuning equals the raw HMC window warmup then the raw scan."""
    init, wkey, keys, scales = setup
    ghmc_key = jax.random.PRNGKey(7)
    _, step_size, inv_mass, _ = _raw_hmc_warmup(init, wkey, scales, N_WARMUP)
    state = _raw_ghmc_init(init, ghmc_key, scales)
    expected = _raw_ghmc_sampling_scan(state, keys, scales, step_size, inv_mass)

    draws, (got_step, got_mass) = run_chain(
        GHMC,
        WINDOW_GHMC,
        init,
        wkey,
        ghmc_key,
        keys,
        _target,
        scales,
        N_WARMUP,
        GHMC_CFG,
    )
    assert np.array_equal(np.asarray(got_step), np.asarray(step_size))
    assert np.array_equal(np.asarray(got_mass), np.asarray(inv_mass))
    actual = (draws.position, draws.divergent)
    for want, have in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def _raw_meads_chains(init, warmup_key, scales, chain_keys):
    """Raw BlackJAX MEADS over a jittered ensemble, then vmapped GHMC scans from its first states.

    The oracle for ``run_chain(GHMC, MEADS, ...)``: it reproduces the key split of the
    ensemble dispersion and the adaptation, and the start from the first ``n_chains``
    warmed states, with alpha and delta taken from the adaptation.
    """
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    def log_density(position):
        return _target(position, scales)

    jitter_key, adapt_key = jax.random.split(warmup_key)
    ensemble = init[None, :] + _shared._MEADS_JITTER_SCALE * jax.random.normal(
        jitter_key, (MEADS_CFG.n_ensemble, N_DIM), dtype=init.dtype
    )
    warmup = blackjax.meads_adaptation(
        log_density,
        num_chains=MEADS_CFG.n_ensemble,
        num_folds=MEADS_CFG.n_folds,
        adaptation_info_fn=get_filter_adapt_info_fn(),
    )
    (last_states, parameters), _ = warmup.run(adapt_key, ensemble, num_steps=MEADS_WARMUP)
    n_chains = chain_keys.shape[0]
    states = jax.tree.map(lambda leaf: leaf[:n_chains], last_states)
    kernel = blackjax.mcmc.ghmc.build_kernel()

    def _chain(state, keys):
        def _step(s, k):
            s, info = kernel(
                k,
                s,
                log_density,
                parameters["step_size"],
                parameters["momentum_inverse_scale"],
                parameters["alpha"],
                parameters["delta"],
            )
            return s, (s.position, info.is_divergent)

        return jax.lax.scan(_step, state, keys)[1]

    positions, divergent = jax.vmap(_chain)(states, chain_keys)
    return positions, divergent, parameters


def test_ghmc_run_chain_meads_matches_raw_ensemble_program(setup):
    """Fused GHMC under MEADS equals the raw ensemble program, output for output.

    The regime must move: most proposals accepted and every chain away from its start,
    so a kernel that swaps alpha and delta changes the draws and fails this test.
    """
    init, wkey, _, _ = setup
    scales = MEADS_SCALES
    chain_keys = jax.random.split(jax.random.PRNGKey(1), MEADS_CHAINS * MEADS_ITER)
    chain_keys = chain_keys.reshape(MEADS_CHAINS, MEADS_ITER, 2)
    positions, divergent, parameters = _raw_meads_chains(init, wkey, scales, chain_keys)

    draws, tuned = run_chain(
        GHMC,
        MEADS,
        init,
        wkey,
        None,
        chain_keys,
        _target,
        scales,
        MEADS_WARMUP,
        MEADS_CFG,
    )
    assert float(np.asarray(draws.divergent).mean()) < 0.5
    assert float(np.asarray(draws.position).std(axis=1).min()) > 1e-3
    assert np.array_equal(np.asarray(draws.position), np.asarray(positions))
    assert np.array_equal(np.asarray(draws.divergent), np.asarray(divergent))
    assert np.array_equal(np.asarray(tuned[0]), np.asarray(parameters["step_size"]))
    assert np.array_equal(np.asarray(tuned[1]), np.asarray(parameters["momentum_inverse_scale"]))
    assert np.array_equal(np.asarray(tuned[2]), np.asarray(parameters["alpha"]))
    assert np.array_equal(np.asarray(tuned[3]), np.asarray(parameters["delta"]))


def test_ghmc_meads_pinned_alpha_and_delta_replace_the_adapted_values(setup):
    """A pinned alpha and delta are the values the sampler and the diagnostics report."""
    init, wkey, keys, scales = setup
    chain_keys = keys[: MEADS_CHAINS * 10].reshape(MEADS_CHAINS, 10, 2)
    pinned = GHMCConfig(alpha=GHMC_ALPHA, delta=GHMC_DELTA, n_ensemble=8, n_folds=4)
    _, tuned = run_chain(
        GHMC, MEADS, init, wkey, None, chain_keys, _target, scales, MEADS_WARMUP, pinned
    )
    assert float(tuned[2]) == pytest.approx(GHMC_ALPHA)
    assert float(tuned[3]) == pytest.approx(GHMC_DELTA)


def test_ghmc_refuses_a_dense_metric():
    """GHMC's momentum generator is diagonal, so a dense request is refused by name."""
    with pytest.raises(ValueError, match="use_dense=True"):
        GHMCConfig(alpha=0.8, delta=0.65, use_dense=True)


def test_ghmc_adaptation_rule_is_meads_for_every_mass_policy():
    """The registered GHMC rule always runs MEADS: its mass policy is inert."""
    from tengri.inference._backend_registry import get_backend

    entry = get_backend("mcmc_ghmc")
    assert entry.chain_program is GHMC
    assert entry.adaptation_for({"dense_mass_matrix": None}) is MEADS
    assert entry.adaptation_for({"dense_mass_matrix": True}) is MEADS


def test_run_chain_refuses_the_meads_ensemble_for_a_non_ghmc_program(setup):
    """The MEADS ensemble state feeds GHMC only; any other program is refused by name."""
    init, wkey, _, scales = setup
    chain_keys = jnp.zeros((2, 4, 2), dtype=jnp.uint32)
    with pytest.raises(ValueError, match="runs only with GHMC, not hmc"):
        run_chain(HMC, MEADS, init, wkey, None, chain_keys, _target, scales, 4, MEADS_CFG)


FIRST_ORDER_TARGET = 0.574
FIRST_ORDER_START = 0.1
FIRST_ORDER_CFG = {
    proposal: FirstOrderConfig(proposal=proposal, target_accept_rate=FIRST_ORDER_TARGET)
    for proposal in ("mala", "barker")
}


def _raw_first_order_kernel(proposal):
    """Raw BlackJAX Barker or MALA kernel with the step size as the only tuned input."""
    if proposal == "barker":
        base = blackjax.mcmc.barker.build_kernel()
        return lambda k, s, ld, step_size: base(k, s, ld, step_size, None)
    base = blackjax.mcmc.mala.build_kernel()
    return lambda k, s, ld, step_size: base(k, s, ld, step_size)


def _raw_first_order_state(proposal, init, scales):
    """Raw BlackJAX Barker or MALA state at ``init``."""
    module = blackjax.mcmc.barker if proposal == "barker" else blackjax.mcmc.mala
    return module.init(init, lambda p: _target(p, scales))


def _raw_first_order_warmup(init, warmup_key, scales, n_warmup, proposal):
    """Raw dual-averaging step-size warmup: the oracle for ``run_chain(FIRST_ORDER, ...)``."""
    from blackjax.adaptation.step_size import dual_averaging_adaptation

    kernel = _raw_first_order_kernel(proposal)

    def log_density(position):
        return _target(position, scales)

    da_init, da_update, da_final = dual_averaging_adaptation(FIRST_ORDER_TARGET)
    state = _raw_first_order_state(proposal, init, scales)
    warmup_keys = jax.random.split(warmup_key, n_warmup)

    def _step(carry, k):
        s, da = carry
        s, info = kernel(k, s, log_density, jnp.exp(da.log_step_size))
        return (s, da_update(da, info.acceptance_rate)), None

    (state, da), _ = jax.lax.scan(_step, (state, da_init(FIRST_ORDER_START)), warmup_keys)
    return state, da_final(da)


def _raw_first_order_sampling_scan(state, chain_keys, scales, step_size, proposal):
    """Raw BlackJAX sampling scan at a fixed step: the oracle for ``sample_chain(FIRST_ORDER)``."""
    kernel = _raw_first_order_kernel(proposal)

    def log_density(position):
        return _target(position, scales)

    def _step(s, k):
        s, info = kernel(k, s, log_density, step_size)
        return s, (s.position, info.acceptance_rate)

    _, outputs = jax.lax.scan(_step, state, chain_keys)
    return outputs


@pytest.mark.parametrize("proposal", ["mala", "barker"])
def test_first_order_sample_chain_matches_raw_scan(setup, proposal):
    """Sampling a first-order chain at a fixed step equals the raw BlackJAX scan."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.05)
    cfg = FIRST_ORDER_CFG[proposal]
    state = FIRST_ORDER.init(init, lambda p: _target(p, scales), None, cfg)

    pos_want, acc_want = _raw_first_order_sampling_scan(state, keys, scales, step_size, proposal)
    got = sample_chain(FIRST_ORDER, state, keys, _target, scales, (step_size, None), cfg)
    assert np.array_equal(np.asarray(got.position), np.asarray(pos_want))
    assert np.array_equal(np.asarray(got.extras["accept"]), np.asarray(acc_want))
    assert got.divergent is None


@pytest.mark.parametrize("proposal", ["mala", "barker"])
def test_first_order_adapt_chain_matches_raw_dual_averaging(setup, proposal):
    """Dual-averaging warmup equals the raw BlackJAX warmup: the step size, nothing else."""
    init, wkey, _keys, scales = setup
    _, want_step = _raw_first_order_warmup(init, wkey, scales, N_WARMUP, proposal)
    (got_step, got_mass), got_divergent, got_n_grad = adapt_chain(
        FIRST_ORDER,
        DUAL_AVERAGING,
        init,
        wkey,
        _target,
        scales,
        N_WARMUP,
        FIRST_ORDER_CFG[proposal],
    )
    assert np.array_equal(np.asarray(got_step), np.asarray(want_step))
    assert got_mass is None
    assert got_divergent is None and got_n_grad is None


@pytest.mark.parametrize("proposal", ["mala", "barker"])
def test_first_order_run_chain_matches_raw_warmup_then_sampling(setup, proposal):
    """Fused first-order equals the raw warmup followed by the raw scan, on the same keys.

    The sampling scan starts from the warmup's final state, as the fused scan always did.
    """
    init, wkey, keys, scales = setup
    state, want_step = _raw_first_order_warmup(init, wkey, scales, N_WARMUP, proposal)
    want_pos, want_acc = _raw_first_order_sampling_scan(state, keys, scales, want_step, proposal)

    draws, params = run_chain(
        FIRST_ORDER,
        DUAL_AVERAGING,
        init,
        wkey,
        None,
        keys,
        _target,
        scales,
        N_WARMUP,
        FIRST_ORDER_CFG[proposal],
    )
    assert np.array_equal(np.asarray(params[0]), np.asarray(want_step))
    assert np.array_equal(np.asarray(draws.position), np.asarray(want_pos))
    assert np.array_equal(np.asarray(draws.extras["accept"]), np.asarray(want_acc))


@pytest.mark.parametrize("proposal", ["mala", "barker"])
def test_first_order_oracle_runs_a_moving_regime(setup, proposal):
    """The parity oracle samples a regime where proposals are accepted and rejected.

    A parity test over a chain that never moves passes for any keys. The mean
    acceptance must sit strictly inside (0, 1) and the chain must move.
    """
    init, wkey, keys, scales = setup
    state, step_size = _raw_first_order_warmup(init, wkey, scales, N_WARMUP, proposal)
    positions, accept = _raw_first_order_sampling_scan(state, keys, scales, step_size, proposal)
    mean_accept = float(np.mean(np.asarray(accept)))
    assert 0.0 < mean_accept < 1.0, f"{proposal}: mean acceptance {mean_accept}"
    assert float(np.std(np.asarray(positions), axis=0).max()) > 1e-3


_RESOLVED_KWARGS = {
    "dense_mass_matrix": False,
    "n_leapfrog_steps": 10,
    "pathfinder_warmstart": False,
}


@pytest.mark.parametrize("path", ["run", "compile"])
def test_every_registered_chain_program_yields_an_adaptation_on_both_paths(path):
    """Every registered MCMC entry with a chain program names an adaptation on each path."""
    programs = [e for e in all_backends() if e.chain_program is not None]
    assert {e.name for e in programs} >= {
        "mcmc_nuts",
        "mcmc_hmc",
        "mcmc_dynamic_hmc",
        "mcmc_ghmc",
        "mcmc_mala",
        "mcmc_barker",
    }
    for entry in programs:
        assert entry.adaptation_for(_RESOLVED_KWARGS, path=path) is not None, entry.name


def test_ghmc_compile_warms_the_window_and_run_samples_with_meads():
    """GHMC names two adaptations, and an unknown path is refused by name."""
    entry = get_backend("mcmc_ghmc")
    assert entry.adaptation_for(_RESOLVED_KWARGS, path="run") is MEADS
    assert entry.adaptation_for(_RESOLVED_KWARGS, path="compile") is WINDOW_GHMC
    with pytest.raises(ValueError, match="path must be"):
        entry.adaptation_for(_RESOLVED_KWARGS, path="sample")


@pytest.mark.parametrize("method", ["mcmc_nuts", "mcmc_hmc", "mcmc_dynamic_hmc", "mcmc_mala"])
def test_compile_warms_the_programs_run_then_hits(method):
    """A run() after compile() adds no adaptation or sampling compilation."""
    fitter = fitter_photometry()
    fitter.compile(
        mcmc_methods=[method],
        modes=(),
        n_warmup=20,
        n_burnin=20,
        n_mcmc_samples=20,
        verbose=False,
    )
    sizes = (adapt_chain._cache_size(), sample_chain._cache_size())
    fitter.run(method, key=jax.random.PRNGKey(0), n_warmup=20, n_burnin=20, n_samples=20)
    assert (adapt_chain._cache_size(), sample_chain._cache_size()) == sizes


def test_dynamic_hmc_compile_warms_the_policy_run_adapts_under_a_dense_basis_spec(monkeypatch):
    """A compile under a dense_basis spec warms the dense metric dynamic HMC run adapts.

    ``run_dynamic_hmc`` defaults to ``dense_mass_matrix=True``; compile must resolve the
    same policy, not the NUTS/HMC ``None`` default.
    """
    from tengri.inference.backends.mcmc import nuts as nuts_backend

    monkeypatch.setattr(nuts_backend, "_spec_uses_dense_basis", lambda spec: True)
    fitter = fitter_photometry()
    fitter.compile(
        mcmc_methods=["mcmc_dynamic_hmc"],
        modes=(),
        n_warmup=20,
        n_burnin=20,
        n_mcmc_samples=20,
        verbose=False,
    )
    sizes = (adapt_chain._cache_size(), sample_chain._cache_size())
    fitter.run(
        "mcmc_dynamic_hmc", key=jax.random.PRNGKey(0), n_warmup=20, n_burnin=20, n_samples=20
    )
    assert (adapt_chain._cache_size(), sample_chain._cache_size()) == sizes


def test_compile_refuses_a_registered_method_without_a_program_by_name():
    """A registered MCMC method with no chain program is refused by name, not warned about."""
    fitter = fitter_photometry()
    with pytest.raises(BackendError, match="mcmc_mclmc"):
        fitter._warm_mcmc_program(
            "mcmc_mclmc",
            init_flat=jnp.zeros(3),
            ld2=None,
            data_args=None,
            n_warmup=1,
            n_chain=1,
        )


# The flat seam's statics for every chain driver and name. Each row is the seam call's
# method and keyword statics, with the ``run_chain`` program, adaptation and cfg that the
# same statics must reproduce. The defaults match run_flat_sampler's; the others move one
# static off its default so the caller's kwargs are shown to reach the chain.
_FLAT_SEAM_CASES = [
    pytest.param("mcmc_nuts", {}, NUTS, WINDOW_NUTS, NUTSConfig(5, False, 0.8), False, id="nuts"),
    pytest.param(
        "pathfinder",
        {},
        NUTS,
        PATHFINDER_NUTS,
        NUTSConfig(5, False, 0.8),
        False,
        id="nuts_pathfinder",
    ),
    pytest.param("mcmc", {}, NUTS, WINDOW_NUTS, NUTSConfig(5, False, 0.8), False, id="mcmc_alias"),
    pytest.param(
        "mcmc_nuts",
        {"dense_mass_matrix": True},
        NUTS,
        WINDOW_NUTS,
        NUTSConfig(5, True, 0.8),
        False,
        id="nuts_dense",
    ),
    pytest.param(
        "mcmc_hmc", {}, HMC, WINDOW_HMC(10, True), HMCConfig(10, False, 0.8), False, id="hmc"
    ),
    pytest.param(
        "mcmc_hmc",
        {"n_leapfrog": 7},
        HMC,
        WINDOW_HMC(7, True),
        HMCConfig(7, False, 0.8),
        False,
        id="hmc_leapfrog7",
    ),
    pytest.param(
        "mcmc_dynamic_hmc",
        {},
        DYNAMIC_HMC,
        WINDOW_HMC(DYNAMIC_HMC_TUNING_LEAPFROG_STEPS, True),
        HMCConfig(DYNAMIC_HMC_TUNING_LEAPFROG_STEPS, False, 0.8),
        True,
        id="dynamic_hmc",
    ),
    pytest.param(
        "mcmc_dynamic_hmc",
        {"dense_mass_matrix": True},
        DYNAMIC_HMC,
        WINDOW_HMC(DYNAMIC_HMC_TUNING_LEAPFROG_STEPS, False),
        HMCConfig(DYNAMIC_HMC_TUNING_LEAPFROG_STEPS, True, 0.8),
        True,
        id="dynamic_hmc_dense",
    ),
    pytest.param(
        "mcmc_ghmc",
        {},
        GHMC,
        WINDOW_GHMC,
        GHMCConfig(alpha=0.8, delta=0.65, target_accept_rate=0.8),
        True,
        id="ghmc",
    ),
    pytest.param(
        "mcmc_ghmc",
        {"ghmc_alpha": 0.5, "ghmc_delta": 0.4},
        GHMC,
        WINDOW_GHMC,
        GHMCConfig(alpha=0.5, delta=0.4, target_accept_rate=0.8),
        True,
        id="ghmc_alpha05",
    ),
]

_FLAT_SEAM_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_FLAT_SEAM_TRUTH = {
    "alpha": 1.5,
    "beta": 1.0,
    "log_total_mass": 10.0,
    "psd_sigma": 1.0,
    "psd_tau_myr": 50.0,
}
_FLAT_SEAM_N = (60, 10, 40)  # n_warmup, n_burnin, n_samples


def _flat_seam_factory(ssp):
    """A stochastic two-component SFH on ``ssp``: the shared PSD pair is always free."""
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    obs = Observation(photometry=Photometry.from_names(_FLAT_SEAM_BANDS))

    def factory(psd_sigma, psd_tau_myr):
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": ["dpl", "field"],
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
                "alpha": Uniform(0.5, 4.0),
                "beta": Uniform(0.3, 3.0),
                "psd_sigma": Uniform(0.1, 4.0, default=float(psd_sigma)),
                "psd_tau_myr": Uniform(1.0, 300.0, default=float(psd_tau_myr)),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )

    return factory


@pytest.fixture(scope="module")
def flat_seam_population(synthetic_ssp_wide_physical):
    """Two galaxies generated from the model at a truth point, so the posterior is proper.

    The data-term fixture (``test_data_term._population_model_factory``) starts the
    flat problem at a NaN log density on this SSP grid, and its chains never move, so
    it cannot tell one key order from another. The physical-peak SSP keeps photometry
    in float64 range, which makes the start point finite and the chains move.
    """

    factory = _flat_seam_factory(synthetic_ssp_wide_physical)
    template = factory(1.0, 50.0)
    truth = {
        name: next(v for suffix, v in _FLAT_SEAM_TRUTH.items() if name.endswith(suffix))
        for name in template.spec.free_params
    }
    flux = np.asarray(template.predict_photometry(truth))
    galaxies = [
        {"flux_obs": flux * (1.0 + 0.02 * i), "noise": np.abs(flux) * 0.05} for i in range(2)
    ]
    return factory, galaxies


@pytest.mark.parametrize("method, statics, program, adaptation, cfg, fresh_init", _FLAT_SEAM_CASES)
def test_flat_seam_draws_equal_run_chain_with_the_same_keys(
    flat_seam_population, method, statics, program, adaptation, cfg, fresh_init
):
    """The hierarchical flat seam feeds run_chain the keys and statics it always did.

    The chain must move (shared-parameter std above 1e-6) and must not be all
    divergent, so that equality is evidence about the sampler rather than about a
    frozen chain. The seam's shared draws must then equal a direct ``run_chain`` call
    with the same warmup, init and chain keys and the same statics.
    """
    from tengri.inference._hierarchical_flat import build_flat_problem, run_flat_sampler
    from tengri.inference.hierarchical import PopulationFitter

    factory, galaxies = flat_seam_population
    key = jax.random.PRNGKey(0)
    n_warmup, n_burnin, n_samples = _FLAT_SEAM_N
    fitter = PopulationFitter(factory, galaxies, _via_routing=True)
    seam = run_flat_sampler(
        fitter,
        method,
        key=key,
        n_warmup=n_warmup,
        n_burnin=n_burnin,
        n_samples=n_samples,
        allow_unvalidated=True,
        verbose=False,
        **statics,
    )
    sigma = np.asarray(seam.shared_samples["psd_sigma"])
    tau = np.asarray(seam.shared_samples["psd_tau_myr"])
    assert min(sigma.std(), tau.std()) > 1e-6, "the chain did not move"
    assert seam.diagnostics["divergent"] < n_samples, "every draw diverged"

    prob = build_flat_problem(fitter, key=key, memory_mode="low", verbose=False)
    wkey, ckey = jax.random.split(jax.random.fold_in(key, 1))
    chain_keys = jax.random.split(ckey, n_burnin + n_samples)
    init_key = jax.random.fold_in(wkey, 1) if fresh_init else None
    draws, _ = run_chain(
        program,
        adaptation,
        prob.init_flat,
        wkey,
        init_key,
        chain_keys,
        prob.log_prob_with_data,
        prob.data_args,
        n_warmup,
        cfg,
    )
    shared = np.asarray(jax.vmap(prob.extract_shared)(draws.position[n_burnin:]))
    assert np.array_equal(sigma, shared[:, 0])
    assert np.array_equal(tau, shared[:, 1])
    divergent = np.asarray(draws.divergent)[n_burnin:]
    assert seam.diagnostics["divergent"] == int(np.sum(divergent))
    assert np.array_equal(seam.diagnostics["divergent_mask"], divergent)


def test_flat_seam_refuses_an_unknown_method_by_name():
    """An unknown flat method is refused by name before any problem is built."""
    from tengri.inference._hierarchical_flat import run_flat_sampler

    with pytest.raises(ValueError, match="mcmc_no_such_driver"):
        run_flat_sampler(None, "mcmc_no_such_driver", key=jax.random.PRNGKey(0))
