# SPDX-License-Identifier: BSD-3-Clause
"""Banded whitening inside the linear-Gaussian marginal, against a dense reference.

The reference whitens with a dense Cholesky factor of the masked covariance and runs the
same closed form in JAX, so the banded substitution is the only thing under test. The
batched check vmaps the one primitive over padded galaxies, which is how the batched
spectroscopy adapter calls it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import solve_triangular

from tengri.observation.banded import BandedMatrix, banded_cholesky, banded_forward_substitution
from tengri.observation.linear_marginal import marginalize_linear_nuisance

pytestmark = pytest.mark.contract

_LOG_2PI = float(np.log(2.0 * np.pi))
_SCALE = 3.0


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


def _masked_cov(cov: np.ndarray, pix_mask: np.ndarray) -> np.ndarray:
    """Pixel-masked covariance M C M + (I - M), the matrix the banded factor factors."""
    m = pix_mask
    return m[:, None] * cov * m[None, :] + np.diag(1.0 - m)


def _dense_marginal(data, model0, design, cov, pix_mask, col_mask, prior_mean, prec):
    """Dense-Cholesky reference of the closed form in ``linear_marginal`` (no flat columns)."""
    n_col = design.shape[1]
    eye = jnp.eye(n_col)
    chol = jnp.linalg.cholesky(cov)
    resid = solve_triangular(chol, pix_mask * (data - model0), lower=True)
    design_w = solve_triangular(chol, (pix_mask[:, None] * design) * _SCALE, lower=True)
    d_mat = jnp.diag(col_mask)
    gram = design_w.T @ design_w + prec * _SCALE**2
    rhs = design_w.T @ resid + _SCALE * (prec @ prior_mean)
    gram_m = d_mat @ gram @ d_mat + (eye - d_mat)
    rhs_m = d_mat @ rhs
    beta_s = jnp.linalg.solve(gram_m, rhs_m)
    cov_s = jnp.linalg.solve(gram_m, eye)
    _sign, logdet_m = jnp.linalg.slogdet(gram_m)
    log_norm = jnp.sum(jnp.log(jnp.diag(chol)))
    log_like = (
        -0.5 * jnp.sum(pix_mask) * _LOG_2PI
        - log_norm
        - 0.5 * resid @ resid
        + 0.5 * rhs_m @ beta_s
        - 0.5 * logdet_m
    )
    prior_unscaled = d_mat @ prec @ d_mat + (eye - d_mat)
    _sign_p, logdet_p = jnp.linalg.slogdet(prior_unscaled)
    cm = col_mask * prior_mean
    log_like = (
        log_like
        + 0.5 * (logdet_p + 2.0 * jnp.sum(col_mask) * jnp.log(_SCALE))
        - 0.5 * cm @ (prec @ cm)
    )
    return log_like, _SCALE * beta_s, _SCALE**2 * cov_s


def _galaxy(seed: int, n: int, k: int, p: int, n_real: int, col_keep: np.ndarray):
    """One toy galaxy: banded covariance, padded tail, interior masked pixels and columns."""
    rng = np.random.default_rng(seed)
    cov = _banded_spd(seed, n, p)
    pix_mask = np.zeros(n)
    pix_mask[:n_real] = 1.0
    pix_mask[3:n_real:17] = 0.0  # interior masked pixels
    data = rng.normal(size=n)
    model0 = 0.1 * rng.normal(size=n)
    design = rng.normal(size=(n, k)) * 5.0
    prior_mean = 0.3 * rng.normal(size=k) * col_keep
    prec = np.diag(1.0 / (0.5 + rng.uniform(size=k)))
    return cov, pix_mask, data, model0, design, col_keep.astype(float), prior_mean, prec


def _banded_marginal(data, model0, design, chol_data, offsets, pix_mask, col_mask, mean, prec):
    chol = BandedMatrix(offsets=jnp.asarray(offsets), data=chol_data)
    return marginalize_linear_nuisance(
        data,
        model0,
        design,
        chol_lower=chol,
        pix_mask=pix_mask,
        prior_mean=mean,
        prior_precision=prec,
        col_mask=col_mask,
        scale=_SCALE,
    )


def test_forward_substitution_on_stacked_rhs_matches_columnwise():
    """One scan over an (n, k) right-hand side equals k separate solves."""
    n, p = 90, 3
    chol = banded_cholesky(_to_banded(_banded_spd(7, n, p), p))
    rhs = jnp.asarray(np.random.default_rng(1).normal(size=(n, 5)))
    stacked = banded_forward_substitution(chol, rhs)
    columnwise = jnp.stack(
        [banded_forward_substitution(chol, rhs[:, j]) for j in range(rhs.shape[1])], axis=1
    )
    np.testing.assert_allclose(np.asarray(stacked), np.asarray(columnwise), rtol=1e-13)


def test_single_galaxy_value_and_gradient_match_dense_reference():
    """Masked pixels, a masked column and a prior mean agree with the dense reference."""
    n, k, p = 120, 4, 3
    col_keep = np.array([1.0, 1.0, 0.0, 1.0])
    cov, pm, data, m0, design, cm, mean, prec = _galaxy(3, n, k, p, 104, col_keep)
    chol = banded_cholesky(_to_banded(cov, p), pm)
    dense_cov = jnp.asarray(_masked_cov(cov, pm))
    pm_j, cm_j = jnp.asarray(pm), jnp.asarray(cm)
    mean_j, prec_j = jnp.asarray(mean), jnp.asarray(prec)

    def banded_value(d, m, x):
        res = _banded_marginal(d, m, x, chol.data, chol.offsets, pm_j, cm_j, mean_j, prec_j)
        return res.log_like, (res.coef_hat, res.coef_cov)

    def dense_value(d, m, x):
        ll, beta, cov_c = _dense_marginal(d, m, x, dense_cov, pm_j, cm_j, mean_j, prec_j)
        return ll, (beta, cov_c)

    args = (jnp.asarray(data), jnp.asarray(m0), jnp.asarray(design))
    grad_b = jax.value_and_grad(banded_value, argnums=(0, 1, 2), has_aux=True)
    grad_d = jax.value_and_grad(dense_value, argnums=(0, 1, 2), has_aux=True)
    (ll_b, (beta_b, cov_b)), g_b = grad_b(*args)
    (ll_d, (beta_d, cov_d)), g_d = grad_d(*args)

    assert float(ll_b) == pytest.approx(float(ll_d), rel=1e-12)
    np.testing.assert_allclose(np.asarray(beta_b), np.asarray(beta_d), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.asarray(cov_b), np.asarray(cov_d), rtol=1e-10, atol=1e-14)
    for gb, gd in zip(g_b, g_d, strict=True):
        scale_ref = np.max(np.abs(np.asarray(gd)))
        np.testing.assert_allclose(
            np.asarray(gb), np.asarray(gd), rtol=1e-10, atol=1e-12 * scale_ref
        )


def test_batched_galaxies_match_dense_reference_per_galaxy():
    """vmap over padded galaxies with per-galaxy masks equals the dense loop, value and grad."""
    n, k, p, n_gal = 96, 3, 2, 3
    real = [96, 80, 88]
    keeps = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
    ]
    gals = [_galaxy(10 + g, n, k, p, real[g], keeps[g]) for g in range(n_gal)]
    chol_data = jnp.stack([banded_cholesky(_to_banded(g[0], p), g[1]).data for g in gals])
    offsets = -np.arange(p + 1)

    def stack(idx):
        return jnp.stack([jnp.asarray(g[idx]) for g in gals])

    pix, cols = stack(1), stack(5)
    data, m0, design = stack(2), stack(3), stack(4)
    mean, prec = stack(6), stack(7)

    def batched_value(d, m, x, cd):
        per_gal = jax.vmap(
            lambda di, mi, xi, ci, pi, coli, mui, prci: (
                _banded_marginal(di, mi, xi, ci, offsets, pi, coli, mui, prci).log_like
            )
        )(d, m, x, cd, pix, cols, mean, prec)
        return jnp.sum(per_gal), per_gal

    def dense_value(d, m, x):
        per_gal = []
        for g in range(n_gal):
            dense_cov = jnp.asarray(_masked_cov(gals[g][0], gals[g][1]))
            ll, _b, _c = _dense_marginal(
                d[g], m[g], x[g], dense_cov, pix[g], cols[g], mean[g], prec[g]
            )
            per_gal.append(ll)
        per_gal = jnp.stack(per_gal)
        return jnp.sum(per_gal), per_gal

    grad_b = jax.value_and_grad(batched_value, argnums=(0, 1, 2), has_aux=True)
    grad_d = jax.value_and_grad(dense_value, argnums=(0, 1, 2), has_aux=True)
    (_s, res_b), g_b = grad_b(data, m0, design, chol_data)
    (_s, res_d), g_d = grad_d(data, m0, design)
    np.testing.assert_allclose(np.asarray(res_b), np.asarray(res_d), rtol=1e-12)
    for gb, gd in zip(g_b[:3], g_d, strict=True):
        scale_ref = np.max(np.abs(np.asarray(gd)))
        np.testing.assert_allclose(
            np.asarray(gb), np.asarray(gd), rtol=1e-10, atol=1e-12 * scale_ref
        )


def test_float32_value_and_gradient_are_finite_and_nonzero():
    """In pure float32 the banded marginal is finite with nonzero value and gradients."""
    n, k, p = 120, 4, 3
    col_keep = np.ones(k)
    cov, pm, data, m0, design, cm, mean, prec = _galaxy(21, n, k, p, 110, col_keep)
    chol = banded_cholesky(_to_banded(cov, p), pm)
    with jax.enable_x64(False):
        cd = jnp.asarray(chol.data, jnp.float32)
        args = tuple(jnp.asarray(a, jnp.float32) for a in (data, m0, design))
        pm32, cm32 = jnp.asarray(pm, jnp.float32), jnp.asarray(cm, jnp.float32)
        mean32, prec32 = jnp.asarray(mean, jnp.float32), jnp.asarray(prec, jnp.float32)

        def value(d, m, x):
            return _banded_marginal(d, m, x, cd, chol.offsets, pm32, cm32, mean32, prec32).log_like

        val, grads = jax.value_and_grad(value, argnums=(0, 1, 2))(*args)
        assert val.dtype == jnp.float32
        assert np.isfinite(float(val))
        for g in grads:
            g_np = np.asarray(g)
            assert np.all(np.isfinite(g_np))
            assert np.linalg.norm(g_np) > 0.0
