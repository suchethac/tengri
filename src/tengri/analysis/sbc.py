# SPDX-License-Identifier: BSD-3-Clause
"""Simulation-based calibration for the two-step population estimator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

__all__ = ["normalized_rank", "run_population_sbc"]


def normalized_rank(log_posterior, grid, truth_sigma, truth_tau_yr):
    """Posterior-mass fraction below the truth, for each shared parameter.

    Parameters
    ----------
    log_posterior: array_like, shape (A * B,)
        Unnormalized log-posterior [nats] on ``grid.nodes``.
    grid: SharedGrid
    truth_sigma: float
        Injected amplitude [dex].
    truth_tau_yr: float
        Injected timescale [yr].

    Returns
    -------
    rank_sigma, rank_tau: float
        Fraction of marginal posterior mass at or below the truth, in ``[0, 1]``.
        For a calibrated posterior these are uniformly distributed.
    """
    a, b = grid.sigma.size, grid.tau_yr.size
    p = np.asarray(jnp.exp(log_posterior - jnp.max(log_posterior))).reshape(a, b)
    p = p / p.sum()
    m_sigma = p.sum(axis=1)
    m_tau = p.sum(axis=0)
    rank_sigma = float(m_sigma[np.asarray(grid.sigma) <= truth_sigma].sum())
    rank_tau = float(m_tau[np.asarray(grid.tau_yr) <= truth_tau_yr].sum())
    return rank_sigma, rank_tau


def run_population_sbc(
    simulate_fn,
    *,
    n_replicates,
    prior_sigma_bounds,
    prior_tau_bounds_yr,
    seed,
    n_sigma=24,
    n_tau=24,
):
    """Rank statistics over replicate populations drawn from the prior.

    Each replicate draws a truth from the prior, asks ``simulate_fn`` for a
    population, runs the two-step estimator, and records where the truth falls
    in the recovered marginal posterior. Calibrated inference gives uniform
    ranks; a posterior that is too narrow piles ranks at the edges, and one that
    is too wide piles them in the middle.

    Parameters
    ----------
    simulate_fn: callable
        ``(sigma_dex, tau_yr, seed) -> (fields, times_yr)``. ``fields`` is
        ``(N, K, n)`` interim centered-field draws [natural-log units] and
        ``times_yr`` is ``(n,)`` [yr]. Injected rather than imported so this
        module depends on no particular simulator: pass the analytic toy to
        calibrate the estimator alone, or the forward model plus the interim
        fit driver to calibrate the whole pipeline.
    n_replicates: int
        Number of replicate populations.
    prior_sigma_bounds: tuple of float
        Amplitude support [dex]; truths are drawn uniformly within it.
    prior_tau_bounds_yr: tuple of float
        Timescale support [yr]; truths are drawn log-uniformly within it.
    seed: int
        NumPy seed for the truth draws and the per-replicate simulator seeds.
    n_sigma, n_tau: int, optional
        Quadrature grid resolution. Default 24 each.

    Returns
    -------
    ranks: dict
        Keys ``"sigma"`` and ``"tau"``, each an ``(n_replicates,)`` float array
        of normalized ranks in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    grid = SharedGrid.uniform(
        sigma_bounds=prior_sigma_bounds,
        tau_bounds_yr=prior_tau_bounds_yr,
        n_sigma=n_sigma,
        n_tau=n_tau,
    )
    out = {"sigma": np.empty(n_replicates), "tau": np.empty(n_replicates)}
    for m in range(n_replicates):
        s_true = float(rng.uniform(*prior_sigma_bounds))
        t_true = float(np.exp(rng.uniform(*np.log(prior_tau_bounds_yr))))
        fields, times_yr = simulate_fn(s_true, t_true, int(rng.integers(0, 2**31 - 1)))
        logp, _ = shared_log_posterior(fields, times_yr, grid)
        r_s, r_t = normalized_rank(logp, grid, s_true, t_true)
        out["sigma"][m], out["tau"][m] = r_s, r_t
    return out
