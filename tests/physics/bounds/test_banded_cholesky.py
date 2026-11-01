# SPDX-License-Identifier: BSD-3-Clause
"""Banded Cholesky factor and banded forward substitution against dense references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_triangular

from tengri.observation.banded import (
    BandedMatrix,
    banded_cholesky,
    banded_forward_substitution,
)

pytestmark = pytest.mark.bounds


def _random_banded_spd(seed: int, n: int, p: int) -> np.ndarray:
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


def _densify_lower(chol: BandedMatrix, n: int) -> np.ndarray:
    """Lower-triangular dense matrix L[i, i + offsets[k]] = data[k, i]."""
    out = np.zeros((n, n))
    offsets = np.asarray(chol.offsets)
    data = np.asarray(chol.data)
    for k, off in enumerate(offsets):
        rows = np.arange(max(0, -off), min(n, n - off))
        out[rows, rows + off] = data[k, rows]
    return out


@pytest.mark.parametrize("n", [50, 200, 400])
@pytest.mark.parametrize("p", [1, 4, 8])
def test_banded_cholesky_matches_numpy(n, p):
    a = _random_banded_spd(seed=n + p, n=n, p=p)
    chol = banded_cholesky(_to_banded(a, p))
    l_ref = np.linalg.cholesky(a)
    np.testing.assert_allclose(
        _densify_lower(chol, n), l_ref, rtol=1e-12, atol=1e-13 * np.abs(l_ref).max()
    )


@pytest.mark.parametrize("n", [50, 400])
@pytest.mark.parametrize("p", [1, 8])
def test_forward_substitution_matches_solve_triangular(n, p):
    a = _random_banded_spd(seed=7 * n + p, n=n, p=p)
    chol = banded_cholesky(_to_banded(a, p))
    l_dense = _densify_lower(chol, n)
    r = np.random.default_rng(3).normal(size=n)
    y = np.asarray(banded_forward_substitution(chol, jnp.asarray(r)))
    y_ref = solve_triangular(l_dense, r, lower=True)
    np.testing.assert_allclose(y, y_ref, rtol=1e-12, atol=1e-13 * np.abs(y_ref).max())


def test_forward_substitution_is_differentiable_and_nonzero():
    a = _random_banded_spd(seed=11, n=60, p=3)
    chol = banded_cholesky(_to_banded(a, 3))
    r = jnp.asarray(np.random.default_rng(5).normal(size=60))

    def loss(v):
        return jnp.sum(banded_forward_substitution(chol, v) ** 2)

    g = np.asarray(jax.grad(loss)(r))
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


def test_non_spd_matrix_is_refused():
    a = _random_banded_spd(seed=2, n=30, p=2)
    a[10, 10] = -5.0
    with pytest.raises(ValueError, match="positive definite"):
        banded_cholesky(_to_banded(a, 2))
