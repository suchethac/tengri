# SPDX-License-Identifier: BSD-3-Clause
"""Banded pixel covariance through the linear-Gaussian marginal primitive."""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.banded import BandedMatrix, banded_cholesky
from tengri.observation.linear_marginal import marginalize_linear_nuisance

pytestmark = pytest.mark.contract

_LOG_2PI = np.log(2.0 * np.pi)


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
    return BandedMatrix(offsets=jnp.asarray(offsets), data=jnp.asarray(data))


def _dense_reference(data, model0, design, cov):
    """Flat-prior marginal likelihood and coefficient posterior from the dense inverse."""
    resid = data - model0
    cinv = np.linalg.inv(cov)
    gram = design.T @ cinv @ design
    rhs = design.T @ cinv @ resid
    beta = np.linalg.solve(gram, rhs)
    _sign, logdet_c = np.linalg.slogdet(cov)
    _sign, logdet_m = np.linalg.slogdet(gram)
    log_like = (
        -0.5 * data.size * _LOG_2PI
        - 0.5 * logdet_c
        - 0.5 * resid @ cinv @ resid
        + 0.5 * rhs @ beta
        - 0.5 * logdet_m
    )
    return log_like, beta, np.linalg.inv(gram)


def _problem(seed: int, n: int, k: int):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=n)
    model0 = rng.normal(size=n) * 0.1
    design = rng.normal(size=(n, k))
    return data, model0, design


def test_diagonal_covariance_equals_sigma_path():
    data, model0, design = _problem(seed=1, n=40, k=3)
    sigma = np.linspace(0.5, 2.0, 40)
    chol = banded_cholesky(np.diag(sigma**2))
    ref = marginalize_linear_nuisance(
        jnp.asarray(data), jnp.asarray(model0), jnp.asarray(design), sigma=jnp.asarray(sigma)
    )
    new = marginalize_linear_nuisance(
        jnp.asarray(data), jnp.asarray(model0), jnp.asarray(design), chol_lower=chol
    )
    np.testing.assert_allclose(new.log_like, ref.log_like, rtol=1e-12)
    np.testing.assert_allclose(new.coef_hat, ref.coef_hat, rtol=1e-12)
    np.testing.assert_allclose(new.coef_cov, ref.coef_cov, rtol=1e-12)


@pytest.mark.parametrize("p", [1, 3])
def test_banded_covariance_matches_dense_reference(p):
    n, k = 60, 2
    data, model0, design = _problem(seed=4 + p, n=n, k=k)
    cov = _banded_spd(seed=9 + p, n=n, p=p)
    got = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.asarray(model0),
        jnp.asarray(design),
        chol_lower=banded_cholesky(_to_banded(cov, p)),
    )
    ref_ll, ref_beta, ref_cov = _dense_reference(data, model0, design, cov)
    np.testing.assert_allclose(got.log_like, ref_ll, rtol=1e-12)
    np.testing.assert_allclose(got.coef_hat, ref_beta, rtol=1e-12)
    np.testing.assert_allclose(got.coef_cov, ref_cov, rtol=1e-12)


def test_masked_pixels_equal_deletion():
    n, k, p = 50, 2, 2
    data, model0, design = _problem(seed=21, n=n, k=k)
    cov = _banded_spd(seed=22, n=n, p=p)
    keep = np.ones(n)
    keep[[5, 6, 17, 31, 32]] = 0.0
    idx = np.flatnonzero(keep)
    masked = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.asarray(model0),
        jnp.asarray(design),
        chol_lower=banded_cholesky(_to_banded(cov, p), pix_mask=keep),
        pix_mask=jnp.asarray(keep),
    )
    deleted = marginalize_linear_nuisance(
        jnp.asarray(data[idx]),
        jnp.asarray(model0[idx]),
        jnp.asarray(design[idx]),
        chol_lower=banded_cholesky(_to_banded(cov[np.ix_(idx, idx)], p)),
    )
    np.testing.assert_allclose(masked.log_like, deleted.log_like, rtol=1e-12)
    np.testing.assert_allclose(masked.coef_hat, deleted.coef_hat, rtol=1e-12)
    np.testing.assert_allclose(masked.coef_cov, deleted.coef_cov, rtol=1e-12)


def test_sigma_and_chol_are_exclusive():
    data, model0, design = _problem(seed=3, n=10, k=1)
    chol = banded_cholesky(np.eye(10))
    with pytest.raises(ValueError, match="exactly one"):
        marginalize_linear_nuisance(
            jnp.asarray(data),
            jnp.asarray(model0),
            jnp.asarray(design),
            sigma=jnp.ones(10),
            chol_lower=chol,
        )
