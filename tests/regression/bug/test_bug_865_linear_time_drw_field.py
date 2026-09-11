# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the ``field`` DRW is stationary in LINEAR (physical) time (#865).

The stochastic ``field`` SFH modulates a smooth baseline by ``exp(gp_x - K0/2)``
where ``gp_x`` is a damped-random-walk Gaussian process. Star-formation
burstiness (gas cycling) is a physical-time process with a fixed decorrelation
timescale, so the DRW covariance is built directly in cosmic time and merely
*sampled* on the (log-spaced) SFH age grid:

    K_ij = (sigma * ln10)^2 * exp(-|t_i - t_j| / tau),   t_i = 10**u_i

Before #865 the GP was instead stationary in LOG-AGE (a Fourier construction on
the log grid with a single reference-time Jacobian), so its correlation length
was a fixed number of dex — scale-free, stretching with cosmic age — and the
``psd_sigma`` / ``psd_tau_myr`` parameters did not deliver the dex amplitude and
physical timescale their priors document (Caplar & Tacchella 2019; Iyer+2020).

This suite pins the corrected behavior (all CI-runnable — no SSP data):

1. ``psd_sigma`` is the modulation std of log10(SFR) in **dex**, directly.
2. The 1/e decorrelation is a **fixed number of Myr** at every age where the
   grid resolves it — NOT a fixed number of dex (no ~10x age stretch).
3. The sigma -> 0 limit recovers no modulation (gp_x -> 0).
4. xi = 0 -> gp_x = 0 (the latent-zero draw is the smooth baseline).
5. The realization is differentiable w.r.t. both psd_sigma and psd_tau (fittable).

References
----------
.. [1] K. G. Iyer et al., MNRAS, 498, 430 (2020). [physical decorrelation]
.. [2] N. Caplar & S. Tacchella, MNRAS, 487, 3845 (2019). [dex amplitude]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import make_log_age_grid
from tengri.components.stellar.sfh.registry import compute_field_gp

pytestmark = pytest.mark.regression_bug

_N_GRID = 256
_LN10 = np.log(10.0)


def _ensemble(psd_sigma_dex, tau_yr, n=300):
    grid = make_log_age_grid(_N_GRID)
    d_log = float(grid[1] - grid[0])
    gp = np.array(
        [
            np.asarray(
                compute_field_gp(
                    jax.random.normal(jax.random.PRNGKey(i), (_N_GRID,)),
                    psd_sigma_dex,
                    tau_yr,
                    _N_GRID,
                    d_log,
                    log_age_grid=grid,
                )[0]
            )
            for i in range(n)
        ]
    )
    return np.asarray(grid), gp


def test_psd_sigma_is_dex_amplitude_directly():
    """std of log10(SFR) modulation equals psd_sigma (dex), independent of tau."""
    for sigma in (0.2, 0.4):
        _, gp = _ensemble(sigma, 150e6)
        dex = gp.std() / _LN10
        assert dex == pytest.approx(sigma, rel=0.12)


def test_decorrelation_is_fixed_in_linear_time():
    """1/e decorrelation is ~fixed in Myr at young and mid ages (not scale-free)."""
    grid, gp = _ensemble(0.3, 150e6)
    t = 10.0**grid
    mod = gp - gp.mean(axis=0)

    def decorr_myr(t0):
        i0 = int(np.argmin(np.abs(t - t0)))
        corr = np.array([np.corrcoef(mod[:, i0], mod[:, j])[0, 1] for j in range(i0, len(t))])
        below = np.nonzero(corr < 1.0 / np.e)[0]
        return (t[i0 + below[0]] - t[i0]) / 1e6 if below.size else np.nan

    dec_young = decorr_myr(30e6)
    dec_mid = decorr_myr(300e6)
    # Both O(tau=150 Myr), and NOT drifting ~10x with age (the old log-age model).
    assert 45.0 < dec_young < 450.0
    assert 45.0 < dec_mid < 450.0
    assert dec_mid / dec_young < 3.0


def test_sigma_zero_limit_has_no_modulation():
    grid = make_log_age_grid(_N_GRID)
    d_log = float(grid[1] - grid[0])
    xi = jax.random.normal(jax.random.PRNGKey(0), (_N_GRID,))
    gp_x, k0_half = compute_field_gp(xi, 1e-8, 150e6, _N_GRID, d_log, log_age_grid=grid)
    assert float(jnp.max(jnp.abs(gp_x))) < 1e-6
    assert float(k0_half) < 1e-12


def test_zero_latent_is_baseline():
    """xi = 0 -> gp_x = 0 (the smooth baseline; deterministic reparametrization)."""
    grid = make_log_age_grid(_N_GRID)
    d_log = float(grid[1] - grid[0])
    gp_x, _ = compute_field_gp(jnp.zeros(_N_GRID), 0.3, 150e6, _N_GRID, d_log, log_age_grid=grid)
    assert np.allclose(np.asarray(gp_x), 0.0, atol=1e-10)


def test_field_gp_is_differentiable_wrt_sigma_and_tau():
    grid = make_log_age_grid(_N_GRID)
    d_log = float(grid[1] - grid[0])
    xi = jax.random.normal(jax.random.PRNGKey(1), (_N_GRID,))

    def summ(sigma, tau):
        gp_x, _ = compute_field_gp(xi, sigma, tau, _N_GRID, d_log, log_age_grid=grid)
        return jnp.sum(gp_x**2)

    g_sigma, g_tau = jax.grad(summ, argnums=(0, 1))(0.3, 150e6)
    assert np.isfinite(float(g_sigma)) and float(g_sigma) != 0.0
    assert np.isfinite(float(g_tau))
    assert np.any(float(g_tau) != 0.0), (
        "`float(g_tau)` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
