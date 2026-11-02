# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for tengri.inference.importance and the Laplace importance check.

Surface protected: ``importance_sample`` (exact log-evidence on a Gaussian,
PSIS diagnostics), ``evaluate_logdensity`` (batched evaluation equals a Python
loop), and ``run_laplace(importance_draws=...)`` end to end: the Student-t
importance check, its reliability warning on a curved target, and the
guarantee that ``importance_draws=0`` leaves the Laplace output untouched.

Expected values come from closed-form Gaussian normalizing constants, never
from re-typed copies of the implementation.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import ImportanceSamplingUnreliableWarning
from tengri.inference.backends._psis import pareto_k_threshold
from tengri.inference.backends.laplace import run_laplace
from tengri.inference.importance import (
    StudentTProposal,
    evaluate_logdensity,
    importance_sample,
)

pytestmark = pytest.mark.contract

_GAUSS_A = np.array(
    [
        [2.0, 0.5, 0.0],
        [0.5, 1.0, 0.2],
        [0.0, 0.2, 1.5],
    ]
)


def _gaussian_logdensity(x, _data_args):
    """Unnormalized ``-0.5 x^T A x``, the same convention as tengri's prior."""
    return -0.5 * x @ jnp.asarray(_GAUSS_A) @ x


def _gaussian_proposal(df=5.0):
    cov = np.linalg.inv(_GAUSS_A)
    return StudentTProposal(
        mean=jnp.zeros(3),
        chol=jnp.asarray(np.linalg.cholesky(cov)),
        df=df,
    )


def _banana_logdensity(x, _data_args):
    """Strongly curved ridge ``x1 = x0^2 / 2 - 2`` with sd 0.2 across it."""
    r = x[1] - 0.5 * x[0] ** 2 + 2.0
    return -0.5 * x[0] ** 2 / 4.0 - 0.5 * 25.0 * r**2


def _unravel_banana(v):
    return {"x0": v[0], "x1": v[1]}


def _identity(p):
    return p


# ── importance_sample: exact Gaussian target ───────────────────────────────


def test_importance_sample_recovers_gaussian_log_evidence():
    """Unnormalized Gaussian with log_norm = -(D/2) log 2pi has log Z = -0.5 log det A."""
    dim = 3
    log_norm = -0.5 * dim * np.log(2.0 * np.pi)
    expected_log_z = -0.5 * np.linalg.slogdet(_GAUSS_A)[1]

    result = importance_sample(
        _gaussian_logdensity,
        None,
        _gaussian_proposal(),
        4000,
        jax.random.PRNGKey(0),
        log_norm=log_norm,
    )

    assert abs(result.log_z - expected_log_z) < 0.02
    assert result.psis.pareto_k < 0.5
    assert result.draws.shape == (4000, dim)
    assert result.log_ratios.shape == (4000,)
    assert 0.0 < result.max_weight_frac <= 1.0


# ── evaluate_logdensity: batching ──────────────────────────────────────────


def test_evaluate_logdensity_matches_python_loop_with_small_batches():
    """Batched evaluation over n=10 with batch_size=3 equals a per-point loop."""
    rng = np.random.default_rng(3)
    positions = jnp.asarray(rng.normal(size=(10, 3)))

    batched = np.asarray(evaluate_logdensity(_gaussian_logdensity, positions, None, batch_size=3))
    looped = np.array([float(_gaussian_logdensity(p, None)) for p in positions])

    assert batched.shape == (10,)
    np.testing.assert_allclose(batched, looped, rtol=0.0, atol=1e-12)


# ── run_laplace: importance check end to end ───────────────────────────────


def _laplace_gaussian(**kwargs):
    return run_laplace(
        key=jax.random.PRNGKey(7),
        logdensity_fn=_gaussian_logdensity,
        data_args=None,
        position=jnp.zeros(3),
        unravel_fn=lambda v: {"a": v[0], "b": v[1], "c": v[2]},
        to_physical_fn=_identity,
        model=None,
        n_samples=500,
        verbose=False,
        **kwargs,
    )


def test_laplace_importance_check_agrees_on_a_gaussian_without_warning():
    """Laplace is exact for a Gaussian, so the importance log Z matches it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        post = _laplace_gaussian(importance_draws=2000)

    assert not [w for w in caught if issubclass(w.category, ImportanceSamplingUnreliableWarning)]
    assert abs(post.diagnostics["log_evidence_is"] - post.diagnostics["log_evidence"]) < 0.02
    assert np.isfinite(post.diagnostics["log_evidence_is_err"])
    assert post.diagnostics["pareto_k"] < pareto_k_threshold(2000)
    assert post.diagnostics["importance_ess"] > 0


def test_laplace_importance_check_warns_on_a_curved_target():
    """A curved ridge makes the Student-t importance ratios heavy-tailed: warn."""
    with pytest.warns(ImportanceSamplingUnreliableWarning):
        post = run_laplace(
            key=jax.random.PRNGKey(11),
            logdensity_fn=_banana_logdensity,
            data_args=None,
            position=jnp.asarray([0.2, -2.0]),
            unravel_fn=_unravel_banana,
            to_physical_fn=_identity,
            model=None,
            n_samples=500,
            importance_draws=4000,
            verbose=False,
        )

    assert post.diagnostics["pareto_k"] > pareto_k_threshold(4000)
    # The proposal is centered at x1 = -2, so unweighted draws average to -2. The
    # ridge's true mean is E[x1] = E[x0^2 / 2] - 2 = 0. The PSIS-reweighted resample
    # must move toward it, which a uniform resample cannot do.
    assert float(np.mean(np.asarray(post.samples["x1"]))) > -1.85


# ── importance_draws=0 is the old Laplace output ───────────────────────────


def test_importance_draws_zero_leaves_samples_bit_identical():
    """importance_draws=0 (the default) changes neither samples nor diagnostic keys."""
    default = _laplace_gaussian()
    explicit_off = _laplace_gaussian(importance_draws=0)

    for name in default.samples:
        np.testing.assert_array_equal(
            np.asarray(default.samples[name]),
            np.asarray(explicit_off.samples[name]),
        )
    assert set(explicit_off.diagnostics) == set(default.diagnostics)
    for key in ("pareto_k", "pareto_k_threshold", "importance_ess", "log_evidence_is"):
        assert key not in explicit_off.diagnostics
