# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2112: chains that move but never mix pass silently.

A 4-chain fit whose chains each jitter by ~1e-3 of the posterior width around
*different* centres, with zero divergences, returned normally. Every guard on
main was blind to it: the convergence check flags only an exact zero range,
the dead-fit guard needs all-divergent draws or an exactly constant column,
and ``refuse_dead_sampling`` needs a divergence fraction of 0.9.

Guard: ``Posterior`` warns :class:`FrozenMovingChainWarning` and sets
``diagnostics["frozen_moving_chains"]`` when a parameter's within-chain spread
is tiny against both the between-chain disagreement and the posterior width.
It warns rather than raising (owner decision for #2112).

Mutation checks:
1. ``test_frozen_chains_warn_and_set_the_flag``: delete the call to the
   detector in ``Posterior.__post_init__`` -> no warning, flag absent.
2. ``test_healthy_mixed_chains_do_not_warn``: drop the within/between gate
   from the detector -> the mixed fit warns.
3. ``test_flag_is_false_for_a_healthy_fit``: set the flag to ``True``
   unconditionally -> the healthy fit reports it.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import FrozenMovingChainWarning
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug

_N_PER_CHAIN = 200
_N_CHAINS = 4


def _posterior(samples):
    return Posterior(
        samples=samples,
        params={"x": jnp.array(0.0)},
        method="mcmc_nuts",
        wall_time_s=1.0,
        diagnostics={
            "n_divergent": 0,
            "n_samples": _N_PER_CHAIN,
            "n_chains": _N_CHAINS,
        },
    )


def _frozen_samples():
    """Four chains, each jittering by 1e-3 around its own centre (unit-scale spread)."""
    rng = np.random.default_rng(2112)
    centres = np.array([-1.0, -0.3, 0.4, 1.2])
    chains = [c + 1e-3 * rng.standard_normal(_N_PER_CHAIN) for c in centres]
    return {"x": jnp.asarray(np.concatenate(chains))}


def _healthy_samples():
    """Four chains that each explore the same unit-scale posterior."""
    rng = np.random.default_rng(7)
    draws = rng.standard_normal(_N_CHAINS * _N_PER_CHAIN)
    return {"x": jnp.asarray(draws)}


def test_frozen_chains_warn_and_set_the_flag():
    with pytest.warns(FrozenMovingChainWarning, match=r"'x'"):
        post = _posterior(_frozen_samples())
    assert post.diagnostics["frozen_moving_chains"] is True


def test_healthy_mixed_chains_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FrozenMovingChainWarning)
        post = _posterior(_healthy_samples())
    assert post.diagnostics["frozen_moving_chains"] is False
