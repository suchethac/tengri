# SPDX-License-Identifier: BSD-3-Clause
"""``metric="dense"`` reaches every sampler: applied, or invariant by mathematics (#3505, M5).

A pilot-covariance whitening is a reparametrization of the target, so it applies to any
sampler. The only samplers that skip it are those whose result is affine-equivariant
(a MAP point, a Laplace Gaussian), recorded as ``invariant`` and pinned here by running
them on a whitened target and mapping back.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.sampling.driver import get_sampler, run_sampler, sampler_names
from tengri.inference.sampling.target import FlatTarget, LatentLayout

pytestmark = pytest.mark.contract

_A = jnp.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [1.0, 1.0, 1.0]])
_Y = jnp.array([0.5, 0.2, 0.4, 0.3])


def _ld(x, args):
    return -0.5 * jnp.sum(((args["y"] - _A @ x) / 0.5) ** 2) - 0.5 * jnp.sum(x**2)


def _target():
    return FlatTarget(_ld, LatentLayout.vector(3), jnp.zeros(3), {"y": _Y})


_FAST = {"n_warmup": 60, "n_burnin": 10, "n_samples": 60, "vi_n_steps": 200, "smc_n_particles": 64}


@pytest.mark.parametrize("name", sorted(sampler_names()))
def test_dense_metric_is_applied_or_invariant_for_every_sampler(name):
    result = run_sampler(
        name, _target(), key=jax.random.PRNGKey(0), n_chains=2, metric="dense", **_FAST
    )
    record = result.diagnostics
    assert record["metric_requested"] == "dense"
    assert record["metric_applied"] in ("pilot", "native", "internal", "invariant")
    assert record["metric_applied"] == (
        "invariant" if get_sampler(name).metric_mode == "invariant" else record["metric_applied"]
    )
    assert bool(jnp.all(jnp.isfinite(result.draws)))


def test_the_diagonal_request_is_recorded_as_diagonal():
    result = run_sampler("nuts", _target(), key=jax.random.PRNGKey(0), **_FAST)
    assert result.diagnostics["metric_applied"] == "diag"


def test_map_is_invariant_under_whitening():
    target = _target()
    direct = run_sampler("map", target, key=jax.random.PRNGKey(0), map_steps=400)
    factor = jnp.array([[1.5, 0.0, 0.0], [0.7, 0.4, 0.0], [-0.3, 0.2, 2.0]])
    whitened, back = target.whiten(jnp.array([0.3, -0.2, 0.1]), factor)
    moved = run_sampler("map", whitened, key=jax.random.PRNGKey(0), map_steps=4000)
    np.testing.assert_allclose(np.asarray(back(moved.draws)), np.asarray(direct.draws), atol=1e-3)


def test_laplace_is_invariant_under_whitening():
    target = _target()
    cfg = {"map_steps": 400, "n_samples": 4000}
    direct = run_sampler("laplace", target, key=jax.random.PRNGKey(1), **cfg)
    factor = jnp.array([[1.5, 0.0, 0.0], [0.7, 0.4, 0.0], [-0.3, 0.2, 2.0]])
    whitened, back = target.whiten(jnp.array([0.3, -0.2, 0.1]), factor)
    moved = back(run_sampler("laplace", whitened, key=jax.random.PRNGKey(2), **cfg).draws)
    flat_direct = np.asarray(direct.draws).reshape(-1, 3)
    flat_moved = np.asarray(moved).reshape(-1, 3)
    np.testing.assert_allclose(flat_moved.mean(0), flat_direct.mean(0), atol=0.03)
    np.testing.assert_allclose(np.cov(flat_moved.T), np.cov(flat_direct.T), atol=0.01)


def test_prior_aware_samplers_keep_the_posterior_when_whitened():
    """ESS, SMC and nested sampling carry the whitened Gaussian prior, not the standard one."""
    target = _target()
    reference = run_sampler(
        "nuts", target, key=jax.random.PRNGKey(0), n_warmup=300, n_samples=2000
    ).draws.reshape(-1, 3)
    for name in ("ess", "smc", "nss"):
        result = run_sampler(
            name,
            target,
            key=jax.random.PRNGKey(3),
            metric="dense",
            n_warmup=100,
            n_samples=2000,
            smc_n_particles=1000,
        )
        draws = result.draws.reshape(-1, 3)
        assert result.diagnostics["metric_applied"] == "pilot"
        np.testing.assert_allclose(draws.mean(0), reference.mean(0), atol=0.05, err_msg=name)
        np.testing.assert_allclose(draws.std(0), reference.std(0), rtol=0.15, err_msg=name)
