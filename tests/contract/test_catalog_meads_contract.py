# SPDX-License-Identifier: BSD-3-Clause
"""The ghmc catalog engine returns every chain of its MEADS ensemble (#3505, M5).

MEADS adapts from an ensemble of at least ``2 D K`` chains and every chain of it is a
sampling chain, so ``n_chains`` is a minimum. Nothing the sampler produced is discarded:
the engine returns ``ghmc_ensemble_size`` chains, which the fitter's live-chain accounting
also reads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.backends.mcmc.catalog import (
    GHMC_DEFAULT_FOLDS,
    GHMC_DEFAULT_SUPERCHAINS,
    _catalog_live_chains,
    build_catalog_mcmc_engine,
    ghmc_default_chains,
    ghmc_ensemble_size,
)

pytestmark = pytest.mark.contract

_DIM = 2


def _ld(x, args):
    return -0.5 * jnp.sum((x - args["mu"]) ** 2)


def _engine(n_chains, **kwargs):
    run_one, _ = build_catalog_mcmc_engine(
        None,
        "ghmc",
        n_warmup=40,
        n_burnin=5,
        n_samples=7,
        n_chains=n_chains,
        flat_logdensity=(_ld, lambda x: x, jnp.zeros(_DIM), {"mu": jnp.zeros(_DIM)}),
        substitute=lambda data, *rest: {"mu": data},
        **kwargs,
    )
    return run_one


def test_a_small_request_returns_the_whole_default_ensemble():
    size = ghmc_ensemble_size(_DIM, 3)
    assert size == ghmc_default_chains(_DIM) and size > 3
    positions, divergent = _engine(3)(
        jnp.zeros(_DIM), jax.random.PRNGKey(0), jnp.ones(_DIM), *([None] * 6)
    )
    assert positions.shape == (size * 7, _DIM)
    assert divergent.shape == (size * 7,)
    assert _catalog_live_chains("ghmc", n_chains=3, n_ensemble="auto", n_dim=_DIM) == size


def test_a_large_request_is_rounded_up_to_the_ensemble_unit():
    import math

    unit = math.lcm(GHMC_DEFAULT_SUPERCHAINS, GHMC_DEFAULT_FOLDS)
    size = ghmc_ensemble_size(_DIM, 33)
    assert size >= 33 and size % unit == 0
    assert size == ghmc_ensemble_size(_DIM, size)


def test_an_explicit_geometry_returns_exactly_n_chains():
    assert ghmc_ensemble_size(_DIM, 16, n_superchains=4, n_folds=4) == 16
    positions, _ = _engine(16, n_superchains=4, n_folds=4)(
        jnp.zeros(_DIM), jax.random.PRNGKey(0), jnp.ones(_DIM), *([None] * 6)
    )
    assert positions.shape == (16 * 7, _DIM)
