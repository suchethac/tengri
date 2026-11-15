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
    # Isotropic part: min(prior scale 1, START_CAP sd) = 0.003, 0.6, 1 per coordinate.
    iso = jnp.array([0.003, 0.6, 1.0])
    scale = start_scale(inverse_mass, 1.0)
    np.testing.assert_allclose(scale**2, OVERDISPERSION**2 * inverse_mass + iso**2)
    dense = jnp.diag(inverse_mass)
    factor = start_scale(dense, 1.0)
    np.testing.assert_allclose(factor @ factor.T, OVERDISPERSION**2 * dense + jnp.diag(iso**2))
    assert start_scale(None, 0.7) == 0.7


def test_the_starts_spread_matches_the_rule():
    inverse_mass = jnp.array([1e-6, 0.04, 1.0])
    iso = np.array([0.003, 0.6, 1.0])
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


@pytest.mark.parametrize("sampler", ["nuts", "hmc"])
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


def _spy_starts(monkeypatch, module):
    """Record ``(centre, scale)`` of every ``jittered_starts`` call, traced or not.

    ``run_chain`` is jitted, so inside it the arguments are tracers; a debug callback reads
    their values at run time. The compile cache is cleared so the spy is traced.
    """
    jax.clear_caches()
    seen = []
    name = "jittered_starts" if hasattr(module, "jittered_starts") else "_jittered_starts"
    real = getattr(module, name)

    def record(base, scale):
        seen.append((np.asarray(base), np.asarray(scale)))

    def spy(base, key, n_chains, scale):
        jax.debug.callback(record, base, jnp.asarray(scale))
        return real(base, key, n_chains, scale)

    monkeypatch.setattr(module, name, spy)
    return seen


def _standard_target(dim=3):
    return FlatTarget(
        lambda x, args: -0.5 * jnp.sum((x / 0.4) ** 2), LatentLayout.vector(dim), jnp.zeros(dim)
    )


@pytest.mark.parametrize("sampler", ["barker", "mala"])
def test_a_family_without_an_adapted_metric_starts_at_the_unadapted_scale(sampler, monkeypatch):
    """Their adaptation returns an identity inverse mass, which is no estimate of the posterior."""
    from tengri.inference.backends.mcmc import chain_program
    from tengri.inference.sampling.chains import UNADAPTED_START_SCALE

    seen = _spy_starts(monkeypatch, chain_program)
    run_sampler(
        sampler,
        _standard_target(),
        key=jax.random.PRNGKey(0),
        n_chains=3,
        n_warmup=50,
        n_samples=20,
        health_warnings=False,
    )
    ((_, scale),) = seen
    np.testing.assert_allclose(scale, UNADAPTED_START_SCALE)


def test_a_family_with_an_adapted_metric_uses_it(monkeypatch):
    from tengri.inference.backends.mcmc import chain_program

    seen = _spy_starts(monkeypatch, chain_program)
    run_sampler(
        "nuts",
        _standard_target(),
        key=jax.random.PRNGKey(0),
        n_chains=3,
        n_warmup=100,
        n_samples=20,
        health_warnings=False,
    )
    ((_, scale),) = seen
    assert scale.shape == (3,) and np.all(scale > OVERDISPERSION * 0.2)


def test_independent_runs_use_the_same_rule(monkeypatch):
    from tengri.inference.sampling import samplers
    from tengri.inference.sampling.chains import UNADAPTED_START_SCALE

    seen = _spy_starts(monkeypatch, samplers)
    run_sampler(
        "map", _standard_target(), key=jax.random.PRNGKey(0), n_chains=3, health_warnings=False
    )
    np.testing.assert_allclose(seen[0][1], UNADAPTED_START_SCALE)
    seen.clear()
    run_sampler(
        "map",
        _standard_target(),
        key=jax.random.PRNGKey(0),
        n_chains=3,
        metric="dense",
        n_samples=40,
        health_warnings=False,
    )
    # The whitened pass: unit posterior scale, so the start is OVERDISPERSION widths.
    np.testing.assert_allclose(seen[-1][1], OVERDISPERSION)


def test_low_rank_warms_up_from_the_centre_and_starts_chains_by_the_same_rule(monkeypatch):
    from tengri.inference.backends.mcmc import _shared
    from tengri.inference.sampling import samplers

    warm_inits = []
    real_warmup = _shared._hmc_low_rank_warmup_only

    def warm_spy(init, *args, **kwargs):
        warm_inits.append(np.asarray(init))
        return real_warmup(init, *args, **kwargs)

    monkeypatch.setattr(_shared, "_hmc_low_rank_warmup_only", warm_spy)
    seen = _spy_starts(monkeypatch, samplers)
    target = FlatTarget(
        lambda x, args: -0.5 * jnp.sum((x / 0.4) ** 2),
        LatentLayout.vector(3),
        jnp.array([0.3, -0.2, 0.1]),
    )
    run_sampler(
        "hmc_lowrank",
        target,
        key=jax.random.PRNGKey(0),
        n_chains=3,
        n_warmup=100,
        n_samples=20,
        health_warnings=False,
    )
    np.testing.assert_array_equal(warm_inits[0], np.array([0.3, -0.2, 0.1]))
    ((centre, scale),) = seen
    np.testing.assert_array_equal(centre, np.array([0.3, -0.2, 0.1]))
    assert scale.shape == (3,), "the adapted low-rank metric sets the start dispersion"
