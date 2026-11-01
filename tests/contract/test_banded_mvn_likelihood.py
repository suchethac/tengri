# SPDX-License-Identifier: BSD-3-Clause
"""Banded covariance on the per-galaxy correlated-Gaussian likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.likelihoods.protocol import MultivariateGaussianLikelihood
from tengri.observation.banded import BandedMatrix, banded_cholesky

pytestmark = pytest.mark.contract


def _banded_spd(seed: int, n: int, p: int) -> np.ndarray:
    """Dense symmetric positive-definite matrix with half-bandwidth p."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    i, j = np.indices((n, n))
    a = np.where(np.abs(i - j) <= p, 0.5 * (a + a.T), 0.0)
    return a + (np.abs(a).sum(axis=1).max() + 1.0) * np.eye(n)


def _to_banded(a: np.ndarray, p: int) -> BandedMatrix:
    """Diagonal-offset storage, data[k, i] = a[i, i + offsets[k]]."""
    n = a.shape[0]
    offsets = np.arange(-p, p + 1)
    data = np.zeros((offsets.size, n))
    for k, off in enumerate(offsets):
        rows = np.arange(max(0, -off), min(n, n - off))
        data[k, rows] = a[rows, rows + off]
    return BandedMatrix(offsets=offsets, data=jnp.asarray(data))


def _old_value(obs, pred, cov_inv) -> float:
    """The dense quadratic form the class computed before the banded branch."""
    diff = obs - pred
    return float(-0.5 * (diff @ cov_inv @ diff))


@pytest.mark.parametrize("p", [1, 3])
def test_banded_likelihood_equals_dense_quadratic_form(p):
    n = 60
    rng = np.random.default_rng(10 + p)
    obs = rng.normal(size=n)
    pred = rng.normal(size=n)
    cov = _banded_spd(seed=20 + p, n=n, p=p)
    chol = banded_cholesky(_to_banded(cov, p))
    likelihood = MultivariateGaussianLikelihood(
        obs=jnp.asarray(obs),
        cov_inv=None,
        chol_data=chol.data,
        chol_offsets=tuple(int(o) for o in chol.offsets),
    )
    got = float(likelihood.log_prob({"spec_fnu": jnp.asarray(pred)}))
    ref = _old_value(obs, pred, np.linalg.inv(cov))
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_diagonal_covariance_equals_sigma_weights():
    n = 40
    rng = np.random.default_rng(31)
    obs = rng.normal(size=n)
    pred = rng.normal(size=n)
    sigma = np.linspace(0.4, 1.8, n)
    chol = banded_cholesky(np.diag(sigma**2))
    likelihood = MultivariateGaussianLikelihood(
        obs=jnp.asarray(obs),
        cov_inv=None,
        chol_data=chol.data,
        chol_offsets=tuple(int(o) for o in chol.offsets),
    )
    got = float(likelihood.log_prob({"spec_fnu": jnp.asarray(pred)}))
    ref = float(-0.5 * np.sum(((obs - pred) / sigma) ** 2))
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_banded_likelihood_gradient_is_finite_and_nonzero():
    n = 30
    rng = np.random.default_rng(41)
    obs = jnp.asarray(rng.normal(size=n))
    chol = banded_cholesky(_to_banded(_banded_spd(seed=42, n=n, p=2), 2))

    def loss(pred):
        likelihood = MultivariateGaussianLikelihood(
            obs=obs,
            cov_inv=None,
            chol_data=chol.data,
            chol_offsets=tuple(int(o) for o in chol.offsets),
        )
        return likelihood.log_prob({"spec_fnu": pred})

    g = np.asarray(jax.grad(loss)(jnp.asarray(rng.normal(size=n))))
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)
