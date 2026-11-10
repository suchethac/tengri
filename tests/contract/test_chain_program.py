# SPDX-License-Identifier: BSD-3-Clause
"""The adapted-chain runner reproduces the NUTS scans bit for bit.

``chain_program`` replaces the per-family scans in ``backends/mcmc/_shared.py``
with one adaptation and one sampling body. Each entry point here is compared with
the scan it replaces on the same keys and the same statics, so a change in key
handling, scan carry or collected outputs shows up as a failed equality rather
than as a shifted posterior. The compiled-program checks pin the shape of the
lowered control flow and the number of compiled entries.

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


def test_sample_chain_matches_nuts_chain_scan(setup):
    """Sampling with fixed adaptation equals ``_nuts_chain_scan`` output for output."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.5)
    inv_mass = scales**2
    state = blackjax.mcmc.nuts.init(init, lambda p: _target(p, scales))

    expected = _shared._nuts_chain_scan(
        state, keys, _target, scales, step_size, inv_mass, CFG.max_doublings
    )
    got = sample_chain(NUTS, state, keys, _target, scales, (step_size, inv_mass), CFG)
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


def test_adapt_chain_matches_nuts_warmup_only(setup):
    """Window adaptation equals ``_nuts_warmup_only``: step size, mass, flags and cost."""
    init, wkey, _, scales = setup
    params, warmup_divergent, warmup_n_grad = adapt_chain(
        NUTS, WINDOW_NUTS, init, wkey, _target, scales, N_WARMUP, CFG
    )
    expected = _shared._nuts_warmup_only(
        init, wkey, _target, scales, N_WARMUP, CFG.use_dense, CFG.target_accept_rate, False, 10
    )
    step_size, inv_mass = params
    for want, have in zip(
        expected, (step_size, inv_mass, warmup_divergent, warmup_n_grad), strict=True
    ):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_pathfinder_adaptation_matches_nuts_warmup_only(setup):
    """Pathfinder adaptation equals ``_nuts_warmup_only`` with the pathfinder flag set."""
    init, wkey, _, scales = setup
    params, warmup_divergent, warmup_n_grad = adapt_chain(
        NUTS, PATHFINDER_NUTS, init, wkey, _target, scales, N_WARMUP, CFG
    )
    expected = _shared._nuts_warmup_only(
        init, wkey, _target, scales, N_WARMUP, CFG.use_dense, CFG.target_accept_rate, True, 10
    )
    step_size, inv_mass = params
    for want, have in zip(
        expected, (step_size, inv_mass, warmup_divergent, warmup_n_grad), strict=True
    ):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_run_chain_matches_nuts_full_scan(setup):
    """The fused warmup-then-sampling program equals ``_nuts_full_scan`` output for output."""
    init, wkey, keys, scales = setup
    draws, params = run_chain(NUTS, WINDOW_NUTS, init, wkey, keys, _target, scales, N_WARMUP, CFG)
    expected = _shared._nuts_full_scan(
        init,
        wkey,
        keys,
        _target,
        scales,
        N_WARMUP,
        CFG.max_doublings,
        CFG.use_dense,
        CFG.target_accept_rate,
        False,
    )
    positions, divergent, expansions, step_size, inv_mass = expected
    step, mass = params
    actual = (draws.position, draws.divergent, draws.extras["expansions"], step, mass)
    for want, have in zip(
        (positions, divergent, expansions, step_size, inv_mass), actual, strict=True
    ):
        assert np.array_equal(np.asarray(have), np.asarray(want))


def test_sample_and_run_lower_to_the_same_control_flow(setup):
    """The sampling entry lowers to the same while ops and lines as ``_nuts_chain_scan``."""
    init, _, keys, scales = setup
    step_size = jnp.asarray(0.5)
    inv_mass = scales**2
    state = blackjax.mcmc.nuts.init(init, lambda p: _target(p, scales))

    ours = sample_chain.lower(NUTS, state, keys, _target, scales, (step_size, inv_mass), CFG)
    theirs = _shared._nuts_chain_scan.lower(
        state, keys, _target, scales, step_size, inv_mass, CFG.max_doublings
    )
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
