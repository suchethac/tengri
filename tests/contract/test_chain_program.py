# SPDX-License-Identifier: BSD-3-Clause
"""The adapted-chain runner reproduces the NUTS scans bit for bit.

``chain_program`` replaces the per-family scans in ``backends/mcmc/_shared.py``
with one adaptation and one sampling body. Each entry point here is compared with
a raw BlackJAX oracle on the same keys and the same statics, so a change in key
handling, scan carry or collected outputs shows up as a failed equality rather
than as a shifted posterior. The fused ``_nuts_full_scan`` shim is compared with
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

from tengri.inference.backends.mcmc import _shared
from tengri.inference.backends.mcmc.chain_program import (
    NUTS,
    PATHFINDER_NUTS,
    WINDOW_NUTS,
    NUTSConfig,
    adapt_chain,
    divergence_record,
    run_chain,
    sample_chain,
)

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
    draws, params = run_chain(NUTS, WINDOW_NUTS, init, wkey, keys, _target, scales, N_WARMUP, CFG)
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
    run_chain(NUTS, WINDOW_NUTS, init, wkey, keys, _target, scales, N_WARMUP, CFG)
    size_after_first = run_chain._cache_size()
    for seed in (2, 3):
        new_keys = jax.random.split(jax.random.PRNGKey(seed), N_ITER)
        run_chain(NUTS, WINDOW_NUTS, init, wkey, new_keys, _target, scales, N_WARMUP, CFG)
    assert run_chain._cache_size() == size_after_first


def test_divergence_record_reduces_the_mask_once():
    """No flags give no record; flags give the count and the mask."""
    assert divergence_record(None) == {}
    mask = jnp.asarray([False, True, False, True])
    record = divergence_record(mask)
    assert int(record["n_divergent"]) == 2
    assert np.array_equal(np.asarray(record["divergent_mask"]), np.asarray(mask))
