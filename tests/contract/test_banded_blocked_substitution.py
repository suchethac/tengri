# SPDX-License-Identifier: BSD-3-Clause
"""Blocked banded forward substitution agrees with the pixel-by-pixel recurrence."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.banded import BandedMatrix, banded_cholesky, banded_forward_substitution

pytestmark = pytest.mark.contract

RTOL = 1e-12
ATOL = 1e-14
MAX_SCAN_BLOCK = 32  # the block size is at least this for every bandwidth tested


def _random_banded_cov(seed: int, n: int, p: int) -> BandedMatrix:
    """Diagonal-offset storage of a random symmetric positive-definite band matrix."""
    rng = np.random.default_rng(seed)
    offsets = np.arange(-p, p + 1)
    data = np.zeros((offsets.size, n))
    for k, off in enumerate(offsets):
        rows = np.arange(max(0, -off), min(n, n - off))
        data[k, rows] = 0.3 * rng.normal(size=rows.size)
    # Diagonal dominance (Gershgorin) makes the band matrix positive definite.
    diag_row = offsets == 0
    band_sum = np.abs(data[~diag_row]).sum(axis=0) * 2.0
    data[diag_row] = band_sum + 1.0
    return BandedMatrix(offsets=jnp.asarray(offsets), data=jnp.asarray(data))


def _scan_reference(chol: BandedMatrix, r: jnp.ndarray) -> jnp.ndarray:
    """Pixel-by-pixel recurrence y_i = (r_i - sum_k L[i, i-k] y_{i-k}) / L[i, i]."""
    if r.ndim == 2:
        return jax.vmap(lambda col: _scan_reference(chol, col), in_axes=1, out_axes=1)(r)
    diag = chol.data[0]
    coef = chol.data[1:].T
    half = coef.shape[1]

    def step(prev, xs):
        r_i, c_i, d_i = xs
        y_i = (r_i - jnp.dot(c_i, prev)) / d_i
        return jnp.concatenate([y_i[None], prev[:-1]]), y_i

    _, y = jax.lax.scan(step, jnp.zeros(half, dtype=jnp.result_type(r, diag)), (r, coef, diag))
    return y


def _cases():
    for n in (37, 512, 4608):
        for p in (1, 3, 80):
            yield pytest.param(n, min(p, n - 1), id=f"n{n}-p{min(p, n - 1)}")


CASES = list(_cases())


def _rhs(seed: int, n: int, shape: str) -> jnp.ndarray:
    rng = np.random.default_rng(seed + 1)
    return jnp.asarray(rng.normal(size=(n,) if shape == "vec" else (n, 24)))


@pytest.mark.parametrize(("n", "p"), CASES)
@pytest.mark.parametrize("shape", ["vec", "mat"])
def test_blocked_matches_scan_reference(n, p, shape):
    chol = banded_cholesky(_random_banded_cov(7, n, p))
    r = _rhs(7, n, shape)
    np.testing.assert_allclose(
        np.asarray(banded_forward_substitution(chol, r)),
        np.asarray(_scan_reference(chol, r)),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(("n", "p"), CASES)
@pytest.mark.parametrize("shape", ["vec", "mat"])
def test_blocked_gradient_matches_scan_reference(n, p, shape):
    chol = banded_cholesky(_random_banded_cov(11, n, p))
    r = _rhs(11, n, shape)
    w = _rhs(12, n, shape)

    def loss(solver):
        return lambda rr: jnp.sum(w * solver(chol, rr) ** 2)

    g_new = jax.grad(loss(banded_forward_substitution))(r)
    g_ref = jax.grad(loss(_scan_reference))(r)
    np.testing.assert_allclose(np.asarray(g_new), np.asarray(g_ref), rtol=RTOL, atol=ATOL)


def test_masked_pixels_are_identity_rows():
    n, p = 512, 3
    mask = np.ones(n)
    mask[[0, 1, 100, 101, 102, 300, n - 1]] = 0.0
    chol = banded_cholesky(_random_banded_cov(5, n, p), pix_mask=mask)
    r = _rhs(5, n, "vec")
    y = np.asarray(banded_forward_substitution(chol, r))
    masked = mask == 0.0
    np.testing.assert_allclose(y[masked], np.asarray(r)[masked], rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("p", [1, 3, 80])
def test_float32_gradient_is_finite_and_nonzero(p):
    n = 4608
    chol64 = banded_cholesky(_random_banded_cov(3, n, p))
    chol = BandedMatrix(offsets=chol64.offsets, data=chol64.data.astype(jnp.float32))
    r = _rhs(3, n, "vec").astype(jnp.float32)

    def loss(rr):
        return jnp.sum(banded_forward_substitution(chol, rr) ** 2)

    y = banded_forward_substitution(chol, r)
    g = jax.grad(loss)(r)
    assert y.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(y)))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0


@pytest.mark.parametrize(("n", "p"), [(4608, 1), (4608, 80), (512, 3)])
def test_scan_length_is_block_count(n, p):
    """The outer scan runs over blocks of pixels, not over every pixel."""
    chol = banded_cholesky(_random_banded_cov(9, n, p))
    r = _rhs(9, n, "vec")
    jaxpr = jax.make_jaxpr(lambda rr: banded_forward_substitution(chol, rr))(r)
    lengths = [e.params["length"] for e in jaxpr.jaxpr.eqns if e.primitive.name == "scan"]
    assert lengths, "expected a lax.scan in banded_forward_substitution"
    assert max(lengths) <= math.ceil(n / MAX_SCAN_BLOCK)
