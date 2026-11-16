# SPDX-License-Identifier: BSD-3-Clause
"""``metric="dense"`` reaches every sampler: applied, or invariant by mathematics.

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


def _ill_conditioned_quadratic():
    """A proper posterior, precision eigenvalues 1.01 to 1e4 + 1, mean two sd from the start."""
    eig = jnp.array([1e-2, 1.0, 1e4])
    rot = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (3, 3)))[0]
    precision = rot @ jnp.diag(eig) @ rot.T + jnp.eye(3)
    mean = rot @ (2.0 / jnp.sqrt(eig + 1.0))
    linear = precision @ mean

    def logdensity(x, args):
        return -0.5 * x @ precision @ x + linear @ x

    return FlatTarget(logdensity, LatentLayout.vector(3), jnp.zeros(3)), mean


def test_map_reaches_the_optimum_dense_where_diagonal_does_not():
    """Adam in the unwhitened coordinates cannot resolve a condition number of 1e4."""
    target, mean = _ill_conditioned_quadratic()
    key = jax.random.PRNGKey(1)
    error = {}
    for metric in ("diag", "dense"):
        result = run_sampler("map", target, key=key, metric=metric, health_warnings=False)
        error[metric] = float(jnp.max(jnp.abs(result.draws[0, 0] - mean)))
    assert error["dense"] < 1e-6, error
    assert error["diag"] > 1e-2, error
    assert result.diagnostics["metric_applied"] == "pilot"


def test_laplace_starts_from_the_map_under_the_requested_metric():
    target, mean = _ill_conditioned_quadratic()
    result = run_sampler(
        "laplace",
        target,
        key=jax.random.PRNGKey(2),
        metric="dense",
        n_samples=4000,
        health_warnings=False,
    )
    draws = np.asarray(result.draws).reshape(-1, 3)
    # The soft direction has sd about 1: 4 sd / sqrt(4000) = 0.063 bounds the Monte Carlo error.
    np.testing.assert_allclose(draws.mean(0), np.asarray(mean), atol=0.063)
    assert result.diagnostics["grad_inf_norm"] < 1e-2


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
        # The posterior sd is 0.41 and 2000 draws from these samplers fix it to about 0.02: the
        # largest gap to the NUTS reference measured at these keys is 0.023 (ESS), 0.017 (SMC)
        # and 0.008 (NSS), so atol = 0.05 is about twice the worst of them.
        np.testing.assert_allclose(draws.std(0), reference.std(0), atol=0.05, err_msg=name)


@pytest.fixture
def spy_sampler():
    """A layered sampler that records the sizes it is asked for and returns constant draws."""
    from tengri.inference._backend_registry import _BACKENDS, register_backend
    from tengri.inference.sampling.driver import Sampler, SamplerResult

    calls = []

    def run(target, key, n_chains, metric, cfg):
        calls.append(cfg["n_samples"])
        return SamplerResult(jnp.ones((n_chains, cfg["n_samples"], target.n_dim)), None, {})

    register_backend("spy", legacy_fitter=False, sampler=Sampler("spy", "chain", "spy", run))(run)
    yield calls
    _BACKENDS.pop("spy")


def test_a_fallback_to_the_diagonal_metric_warns_once_with_the_reason(spy_sampler):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_sampler(
            "spy",
            _target(),
            key=jax.random.PRNGKey(0),
            metric="dense",
            n_samples=10,
            health_warnings=False,
        )
    fell = [str(w.message) for w in caught if "fell back" in str(w.message)]
    assert len(fell) == 1
    assert "usable draws for D=3" in fell[0] and "D + 1 = 4" in fell[0]
    assert result.diagnostics["metric_fallback"] is True


def test_the_pilot_is_sized_from_d(spy_sampler):
    from tengri.inference.sampling.driver import pilot_rows

    assert pilot_rows(3) == 16
    run_sampler(
        "spy",
        _target(),
        key=jax.random.PRNGKey(0),
        metric="dense",
        n_samples=5,
        health_warnings=False,
    )
    # The pilot asked for more than the 5 draws requested; the final run asked for 5.
    assert spy_sampler[0] == pilot_rows(3) and spy_sampler[1] == 5


def test_a_point_estimate_pilots_with_the_curvature_not_a_sampler(monkeypatch):
    """Whitening a MAP or a Laplace must not run a NUTS warmup just to learn a covariance."""
    import dataclasses

    from tengri.inference._backend_registry import _BACKENDS
    from tengri.inference.sampling.driver import get_sampler

    ran = []
    nuts = get_sampler("nuts")

    def spy(*args, **kwargs):
        ran.append("nuts")
        return nuts.run(*args, **kwargs)

    spied = dataclasses.replace(nuts, run=spy)
    for method in ("mcmc", "mcmc_nuts"):
        monkeypatch.setitem(
            _BACKENDS, method, dataclasses.replace(_BACKENDS[method], sampler=spied)
        )
    target, _mean = _ill_conditioned_quadratic()
    for name in ("map", "laplace"):
        result = run_sampler(
            name,
            target,
            key=jax.random.PRNGKey(2),
            metric="dense",
            n_samples=200,
            health_warnings=False,
        )
        assert result.diagnostics["metric_fallback"] is False
    assert ran == []
