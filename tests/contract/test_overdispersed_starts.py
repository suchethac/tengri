# SPDX-License-Identifier: BSD-3-Clause
"""The n_chains layer starts chains overdispersed, so R-hat can detect multimodality (#3505).

The start covariance is ``OVERDISPERSION^2 Sigma_adapted + diag(s^2)`` with ``s =
min(prior scale, START_CAP sd)``: at least two adapted posterior standard deviations on every
coordinate, and up to the prior's scale where the posterior is not much narrower.
A start dispersion of 0.1 against a posterior sd of 0.43 (the fixture's) made a cross-chain
R-hat a consistency check.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.sampling.chains import OVERDISPERSION, jittered_starts, start_scale
from tengri.inference.sampling.driver import run_sampler
from tengri.inference.sampling.target import FlatTarget, LatentLayout

pytestmark = pytest.mark.contract

_A = jnp.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [1.0, 1.0, 1.0]])
_PREC = np.asarray(_A.T @ _A) / 0.25 + np.eye(3)
_SD = np.sqrt(np.diag(np.linalg.inv(_PREC)))


def _gauss(x, args):
    return -0.5 * jnp.sum(((args["y"] - _A @ x) / 0.5) ** 2) - 0.5 * jnp.sum(x**2)


def test_the_start_covariance_is_the_stated_rule():
    inverse_mass = jnp.array([1e-6, 0.04, 1.0])
    # Isotropic part: min(prior scale 1, START_CAP sd) = 0.01, 1, 1 per coordinate.
    iso = jnp.array([0.01, 1.0, 1.0])
    scale = start_scale(inverse_mass, 1.0)
    np.testing.assert_allclose(scale**2, OVERDISPERSION**2 * inverse_mass + iso**2)
    dense = jnp.diag(inverse_mass)
    factor = start_scale(dense, 1.0)
    np.testing.assert_allclose(factor @ factor.T, OVERDISPERSION**2 * dense + jnp.diag(iso**2))
    assert start_scale(None, 0.7) == 0.7


def test_the_starts_spread_matches_the_rule():
    inverse_mass = jnp.array([1e-6, 0.04, 1.0])
    iso = np.array([0.01, 1.0, 1.0])
    starts = jittered_starts(
        jnp.zeros(3), jax.random.PRNGKey(0), 20000, start_scale(inverse_mass, 1.0)
    )
    np.testing.assert_allclose(
        np.asarray(starts).std(0),
        np.sqrt(OVERDISPERSION**2 * np.asarray(inverse_mass) + iso**2),
        rtol=0.03,
    )


def test_the_chains_of_the_sweep_fixture_start_wider_than_the_posterior():
    """First draw of 64 chains, from starts the layer made: spread over the posterior sd."""
    target = FlatTarget(
        _gauss,
        LatentLayout.vector(3),
        jnp.zeros(3),
        {"y": jnp.array([0.5, 0.2, 0.4, 0.3])},
    )
    result = run_sampler(
        "hmc",
        target,
        key=jax.random.PRNGKey(0),
        n_chains=64,
        n_warmup=300,
        n_burnin=0,
        n_samples=2,
        health_warnings=False,
    )
    spread = np.asarray(result.draws[:, 0, :]).std(0) / _SD
    assert np.all(spread >= 1.0), spread


def _bimodal(x, args):
    """Two modes at +-1.2 on the first coordinate, 0.3 wide: 8 widths apart, inside the prior."""
    lo = -0.5 * ((x[0] + 1.2) / 0.3) ** 2
    hi = -0.5 * ((x[0] - 1.2) / 0.3) ** 2
    return jnp.logaddexp(lo, hi) - 0.5 * jnp.sum(x[1:] ** 2)


def _run_bimodal(sampler):
    target = FlatTarget(_bimodal, LatentLayout.vector(2), jnp.array([1.2, 0.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_sampler(
            sampler,
            target,
            key=jax.random.PRNGKey(2),
            n_chains=8,
            n_warmup=300,
            n_samples=400,
        )
    return result, [str(w.message) for w in caught if "R-hat" in str(w.message)]


@pytest.mark.parametrize("sampler", ["nuts", "hmc", "mala"])
def test_rhat_detects_a_bimodal_target(sampler):
    result, warned = _run_bimodal(sampler)
    assert result.diagnostics["rhat_max"] > 1.1, result.diagnostics["rhat_max"]
    assert len(warned) == 1 and "'x[0]'" in warned[0]


def test_rhat_stays_quiet_on_the_unimodal_control():
    target = FlatTarget(
        lambda x, args: -0.5 * jnp.sum((x / 0.4) ** 2), LatentLayout.vector(2), jnp.zeros(2)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_sampler(
            "nuts", target, key=jax.random.PRNGKey(2), n_chains=8, n_warmup=300, n_samples=400
        )
    assert not [w for w in caught if "R-hat" in str(w.message)]
    assert result.diagnostics["rhat_max"] < 1.1
