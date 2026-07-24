# SPDX-License-Identifier: BSD-3-Clause
"""Regression: field DRW uses the OU state-space (innovations) square root (#1301).

Post-#1271 the GP field latents reach the likelihood, so the field posterior is
genuinely 25-D and HMC diverges: the dense-Cholesky square root ``s = L(sigma,tau)
xi`` is a *dense rotation* that re-orients as ``psd_tau`` moves, so the target's
principal axes rotate with a sampled hyperparameter and a single global mass matrix
cannot track it (up to 864 divergences / 8000 draws, max R-hat 1.16).

The fix replaces the dense-Cholesky square root with the exact OU forward recursion

    s_0 = sqrt(var) xi_0
    s_i = rho_i s_{i-1} + sqrt(var (1 - rho_i^2)) xi_i,   rho_i = exp(-dt_i / tau)

with ``var = (sigma ln10)^2`` and ``dt_i = t_i - t_{i-1}`` on the physical-time grid
``t_i = 10**u_i``. Because a damped random walk *is* an exact first-order Markov
process, this recursion reproduces the SAME covariance ``K(sigma,tau)`` as the dense
Cholesky -- it is a *bit-exact-same-prior* reparameterization -- but ``tau`` now
enters the ``xi -> SFH`` map only through the per-step scalars ``rho_i`` (banded,
local) instead of a dense rotation, which HMC can track.

This suite pins the reparameterization (all CI-runnable -- no SSP data):

1. The innovations map's implied covariance equals the exact DRW ``K`` (same prior).
2. ``xi = 0 -> gp_x = 0`` (deterministic; the latent-zero draw is the baseline).
3. The lognormal bias correction is ``k0_half = var/2`` (marginal variance preserved).
4. The realization is JIT/grad/vmap-safe and differentiable w.r.t. sigma and tau.

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


@pytest.mark.parametrize("n_grid", [16, 32, 64])
@pytest.mark.parametrize("sigma", [0.3, 0.8])
@pytest.mark.parametrize("tau_yr", [1e7, 1e8, 5e8])
def test_innovations_covariance_matches_exact_drw_K(n_grid, sigma, tau_yr):
    """Same-prior guard: implied covariance == exact DRW K to ~machine precision."""
    grid = make_log_age_grid(n_grid)
    K = _exact_drw_K(grid, sigma, tau_yr)
    cov = _implied_covariance(sigma, tau_yr, grid)
    var = (sigma * _LN10) ** 2
    # Relative to the process variance; tolerance far tighter than any modelling
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
