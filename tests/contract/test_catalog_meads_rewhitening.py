# SPDX-License-Identifier: BSD-3-Clause
"""MEADS ("ghmc") in the catalog engine: MAP whitening plus ensemble re-whitening.

The target is a four-dimensional correlated Gaussian whose covariance has condition
number 1e4 and whose smallest covariance eigenvalue-inverse (precision) lies below the
analytic metric's floor, so the metric at the MAP is only approximately right. The engine
must recover the mean and covariance to Monte Carlo error, the adapted step size must stay
usable, and an ensemble with fewer than two chains per sampled coordinate in each fold must
be refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import ConfigError
from tengri.inference.backends.mcmc._shared import ghmc_catalog_scan
from tengri.inference.backends.mcmc.catalog import (
    build_catalog_mcmc_engine,
    ghmc_superchain_starts,
)

pytestmark = pytest.mark.contract

_DIM = 4
_N_FOLDS = 4
_N_SUPERCHAINS = 8
_N_CHAINS = 2 * _DIM * _N_FOLDS
_N_WARMUP = 1000
_N_SAMPLES = 800
_SIGMA_LEVEL = 4.0

_COV_EIGENVALUES = np.array([4.0, 0.3, 1.0e-2, 4.0e-4])
_MEAN = np.array([1.0, -2.0, 0.5, 3.0])


def _rotation() -> np.ndarray:
    """A fixed orthogonal matrix, so the eigen-directions are not axis aligned."""
    rng = np.random.default_rng(11)
    q, _ = np.linalg.qr(rng.standard_normal((_DIM, _DIM)))
    return q


def _truth() -> tuple[np.ndarray, np.ndarray]:
    """Covariance and precision of the target."""
    q = _rotation()
    cov = (q * _COV_EIGENVALUES) @ q.T
    prec = (q / _COV_EIGENVALUES) @ q.T
    return 0.5 * (cov + cov.T), 0.5 * (prec + prec.T)


def _logdensity():
    """Gaussian log-density with the module's mean and precision."""
    _, prec = _truth()
    prec_j, mean_j = jnp.asarray(prec), jnp.asarray(_MEAN)

    def logdensity(x, data_args):
        d = x - mean_j
        return -0.5 * d @ prec_j @ d

    return logdensity


def _substitute(data, noise, presence, redshift, line_flux_obs, line_flux_err, shared):
    """No per-galaxy data: the target is fully specified by the log-density."""
    return {}


def _engine(**overrides):
    """Build the ghmc catalog engine on the toy target."""
    kwargs = dict(
        n_warmup=_N_WARMUP,
        n_burnin=100,
        n_samples=_N_SAMPLES,
        n_chains=_N_CHAINS,
        n_superchains=_N_SUPERCHAINS,
        n_folds=_N_FOLDS,
        flat_logdensity=(_logdensity(), lambda x: x, jnp.asarray(_MEAN), {}),
        substitute=_substitute,
    )
    kwargs.update(overrides)
    run_one, _ = build_catalog_mcmc_engine(None, "ghmc", **kwargs)
    return run_one


@pytest.fixture(scope="module")
def draws() -> np.ndarray:
    """Draws of shape (n_chains, n_samples, D) from the default schedule."""
    run_one = _engine()
    positions, divergent = jax.jit(run_one)(
        jnp.asarray(_MEAN), jax.random.PRNGKey(0), None, None, None, None, None, None, {}
    )
    assert not bool(jnp.any(divergent))
    return np.asarray(positions).reshape(_N_CHAINS, _N_SAMPLES, _DIM)


def test_mean_recovered_to_monte_carlo_error(draws):
    chain_means = draws.mean(axis=1)
    mean = chain_means.mean(axis=0)
    stderr = chain_means.std(axis=0, ddof=1) / np.sqrt(_N_CHAINS)
    assert np.all(stderr > 0.0)
    assert np.all(np.abs(mean - _MEAN) <= _SIGMA_LEVEL * stderr), (mean - _MEAN, stderr)


def test_covariance_recovered_to_monte_carlo_error(draws):
    cov_true, _ = _truth()
    per_chain = np.stack([np.cov(chain, rowvar=False) for chain in draws])
    cov = per_chain.mean(axis=0)
    stderr = per_chain.std(axis=0, ddof=1) / np.sqrt(_N_CHAINS)
    assert np.all(stderr > 0.0)
    assert np.all(np.abs(cov - cov_true) <= _SIGMA_LEVEL * stderr), (cov - cov_true, stderr)


def test_adapted_step_size_is_finite_and_usable_from_a_raw_start():
    """The re-whitening alone (no engine whitening) recovers a usable adapted step size."""
    starts = ghmc_superchain_starts(
        jnp.asarray(_MEAN), jax.random.PRNGKey(2), _N_CHAINS, _N_SUPERCHAINS, 0.05
    )
    chain_keys = jax.random.split(jax.random.PRNGKey(3), _N_CHAINS * 50).reshape(_N_CHAINS, 50, 2)
    _, _, step_size, momentum_inv_scale, alpha, delta = ghmc_catalog_scan(
        starts, jax.random.PRNGKey(4), chain_keys, _logdensity(), {}, _N_WARMUP, _N_FOLDS, 0.5
    )
    assert bool(jnp.isfinite(step_size)) and float(step_size) > 1e-6
    assert bool(jnp.all(jnp.isfinite(momentum_inv_scale)))
    assert bool(jnp.all(momentum_inv_scale > 0.0))
    assert bool(jnp.isfinite(alpha)) and bool(jnp.isfinite(delta))


def test_too_few_chains_per_fold_is_refused_with_the_minimum_named():
    with pytest.raises(ConfigError, match=r"n_chains >= 32") as err:
        _engine(n_chains=16, n_superchains=4)
    assert "outlying chain" in str(err.value)


def test_float32_draws_are_finite():
    with jax.enable_x64(False):
        mean32 = jnp.asarray(_MEAN, dtype=jnp.float32)
        run_one = _engine(
            flat_logdensity=(_logdensity(), lambda x: x, mean32, {}),
            n_warmup=300,
            n_samples=50,
        )
        positions, _ = run_one(
            mean32, jax.random.PRNGKey(5), None, None, None, None, None, None, {}
        )
    assert positions.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(positions)))
    assert float(jnp.std(positions)) > 0.0
