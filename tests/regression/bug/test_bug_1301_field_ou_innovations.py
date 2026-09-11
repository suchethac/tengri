# SPDX-License-Identifier: BSD-3-Clause
"""Regression: field DRW square root is the exact O(n) OU innovations recursion.

The field DRW realization ``s = L(sigma,tau) xi`` is computed by the Ornstein-
Uhlenbeck forward recursion

    s_0 = sqrt(var) xi_0
    s_i = rho_i s_{i-1} + sqrt(var (1 - rho_i^2)) xi_i,   rho_i = exp(-dt_i / tau)

with ``var = (sigma ln10)^2`` and ``dt_i = t_i - t_{i-1}`` on the physical-time grid
``t_i = 10**u_i``, replacing a dense ``jnp.linalg.cholesky`` of the (n, n) covariance.

**This is not a different square root -- it is the same one, computed better.**
Unrolling the recursion gives ``s = M xi`` with ``M`` lower-triangular and positive
on the diagonal (``M_00 = sqrt(var)``, ``M_ii = sqrt(var (1 - rho_i^2))``), and
``M M^T = K``. The Cholesky factor is the *unique* lower-triangular matrix with
positive diagonal satisfying ``L L^T = K``, so ``M == L`` exactly -- a consequence of
a damped random walk being an exact first-order Markov process. Verified below to
~1e-14 against an un-jittered Cholesky.

What the change therefore does and does not buy:

* **Does**: O(n) instead of O(n^3); no positive-definiteness jitter, so the prior is
  the exact ``K`` rather than ``K + 1e-6 var I`` (the old path perturbed every
  realization by ~1e-5 relative); no dense (n, n) intermediate.
* **Does NOT**: change the posterior geometry. The ``xi -> SFH`` map is numerically
  identical, so this is *not* a fix for the #1301 HMC divergences and cannot be --
  any exact square root of ``K(tau)`` carries the same ``tau``-dependence. Escaping
  that requires changing the representation (the linear-time Fourier basis, #1333),
  which changes the prior.

This suite pins the realization (all CI-runnable -- no SSP data):

1. The map equals the un-jittered dense Cholesky factor to machine precision, and
   the induced matrix is lower-triangular with a positive diagonal.
2. The implied covariance equals the exact DRW ``K`` (same prior).
3. ``xi = 0 -> gp_x = 0`` (deterministic; the latent-zero draw is the baseline).
4. The lognormal bias correction is ``k0_half = var/2`` (marginal variance preserved).
5. The realization is JIT/grad/vmap-safe and differentiable w.r.t. sigma and tau.

References
----------
.. [1] K. G. Iyer et al., MNRAS, 498, 430 (2020). [physical decorrelation timescale]
.. [2] N. Caplar & S. Tacchella, MNRAS, 487, 3845 (2019). [PSD amplitude, dex]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import make_log_age_grid

# Imported symbol does not exist yet -> this module fails to import (RED) until
# ``drw_innovations_gp_from_xi`` is implemented in gp_sfh.py.
from tengri.components.stellar.sfh.gp_sfh import drw_innovations_gp_from_xi

pytestmark = pytest.mark.regression_bug

_LN10 = float(np.log(10.0))


def _exact_drw_K(log_age_grid, sigma_dex, tau_yr):
    """The exact linear-time DRW covariance the innovations map must reproduce."""
    t = 10.0 ** np.asarray(log_age_grid)
    var = (sigma_dex * _LN10) ** 2
    dt = np.abs(t[:, None] - t[None, :])
    return var * np.exp(-dt / tau_yr)


def _implied_covariance(sigma_dex, tau_yr, grid):
    """Implied covariance of the innovations map.

    The map ``gp_x = A @ xi`` is linear in ``xi``, so its columns are recovered by
    applying it to the identity basis: ``A[:, j] = map(e_j)`` (``map(0) = 0``). The
    implied covariance is then ``A A^T``.
    """
    n = grid.shape[0]

    def col(e):
        return drw_innovations_gp_from_xi(e, sigma_dex, tau_yr, grid)[0]

    A = jax.vmap(col)(jnp.eye(n)).T  # columns are map(e_j)
    return np.asarray(A @ A.T)


def _innovations_matrix(sigma_dex, tau_yr, grid):
    """The dense matrix ``M`` induced by the innovations map (columns ``map(e_j)``)."""
    n = grid.shape[0]

    def col(e):
        return drw_innovations_gp_from_xi(e, sigma_dex, tau_yr, grid)[0]

    return np.asarray(jax.vmap(col)(jnp.eye(n)).T)


@pytest.mark.parametrize("n_grid", [16, 32, 64])
@pytest.mark.parametrize("sigma", [0.3, 0.8])
@pytest.mark.parametrize("tau_yr", [1e7, 1e8, 5e8])
def test_innovations_is_exactly_the_drw_cholesky_factor(n_grid, sigma, tau_yr):
    """The recursion computes the Cholesky factor itself, not merely a co-square-root.

    A same-covariance check alone would permit any right-multiplication by an
    orthogonal matrix (a different realization of the same prior). This pins the
    stronger, true invariant: the *map* is identical, so no posterior geometry
    changed -- only the cost of computing it and the removal of the jitter.
    """
    grid = make_log_age_grid(n_grid)
    xi = jax.random.normal(jax.random.PRNGKey(7), (n_grid,))
    got = np.asarray(drw_innovations_gp_from_xi(xi, sigma, tau_yr, grid)[0])
    # Exact K with NO positive-definiteness jitter -- the mathematical reference.
    # ``exp(-dt/tau)`` underflows to zero where the grid step far exceeds tau (an
    # exact, expected result); errstate keeps that from surfacing as a warning on
    # the next numpy call, which reads the shared FPU flags.
    with np.errstate(all="ignore"):
        L = np.linalg.cholesky(_exact_drw_K(grid, sigma, tau_yr))
        residual = np.max(np.abs(got - L @ np.asarray(xi)))
    assert np.all(np.isfinite(L))
    assert residual / (sigma * _LN10) < 1e-12  # relative to the process std


def test_innovations_matrix_is_lower_triangular_with_positive_diagonal():
    """The structural reason ``M == L``: Cholesky factors with this shape are unique."""
    grid = make_log_age_grid(32)
    M = _innovations_matrix(0.5, 1e8, grid)
    assert np.max(np.abs(np.triu(M, 1))) == 0.0
    assert bool(np.all(np.diag(M) > 0.0))


@pytest.mark.parametrize("n_grid", [16, 32, 64])
@pytest.mark.parametrize("sigma", [0.3, 0.8])
@pytest.mark.parametrize("tau_yr", [1e7, 1e8, 5e8])
def test_innovations_covariance_matches_exact_drw_K(n_grid, sigma, tau_yr):
    """Same-prior guard: implied covariance == exact DRW K to ~machine precision."""
    grid = make_log_age_grid(n_grid)
    K = _exact_drw_K(grid, sigma, tau_yr)
    cov = _implied_covariance(sigma, tau_yr, grid)
    var = (sigma * _LN10) ** 2
    # Relative to the process variance; tolerance far tighter than any modeling
    # difference -- this asserts the two square roots share the SAME Gram matrix.
    assert np.max(np.abs(cov - K)) / var < 1e-10


def test_innovations_zero_latent_is_baseline():
    """xi = 0 -> gp_x = 0 (deterministic reparametrization; the smooth baseline)."""
    grid = make_log_age_grid(64)
    gp_x, _ = drw_innovations_gp_from_xi(jnp.zeros(64), 0.3, 150e6, grid)
    assert np.allclose(np.asarray(gp_x), 0.0, atol=1e-10)


def test_innovations_k0_half_is_half_variance():
    """Lognormal bias correction preserves the marginal variance: k0_half = var/2."""
    grid = make_log_age_grid(64)
    sigma = 0.4
    _, k0_half = drw_innovations_gp_from_xi(jnp.zeros(64), sigma, 150e6, grid)
    assert float(k0_half) == pytest.approx(0.5 * (sigma * _LN10) ** 2, rel=1e-10)


def test_innovations_is_jit_and_vmap_safe():
    grid = make_log_age_grid(64)
    xis = jax.random.normal(jax.random.PRNGKey(0), (5, 64))

    def one(xi):
        return drw_innovations_gp_from_xi(xi, 0.3, 150e6, grid)[0]

    batch = jax.jit(jax.vmap(one))(xis)
    assert batch.shape == (5, 64)
    assert bool(jnp.all(jnp.isfinite(batch)))


def test_innovations_is_differentiable_wrt_sigma_and_tau():
    grid = make_log_age_grid(64)
    xi = jax.random.normal(jax.random.PRNGKey(1), (64,))

    def summ(sigma, tau):
        gp_x, _ = drw_innovations_gp_from_xi(xi, sigma, tau, grid)
        return jnp.sum(gp_x**2)

    g_sigma, g_tau = jax.grad(summ, argnums=(0, 1))(0.3, 150e6)
    assert np.isfinite(float(g_sigma)) and float(g_sigma) != 0.0
    assert np.isfinite(float(g_tau))
    assert np.any(float(g_tau) != 0.0), (
        "`float(g_tau)` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
