# SPDX-License-Identifier: BSD-3-Clause
"""Linear-Gaussian toy population with an analytic shared posterior.

The observation operator is the identity, so a galaxy's data is its centered
field plus white noise. That makes the per-galaxy marginal likelihood a plain
multivariate normal and the shared posterior exactly computable on a grid --
which is what lets an estimator bug be told apart from an inference bug.

The field mean carries the lognormal bias term ``-sigma_s**2 / 2`` on purpose:
it is the term a careless reconstruction drops, and dropping it must make these
tests fail rather than pass with a small bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

__all__ = ["ToyPopulation", "closed_form_log_posterior", "field_mean", "make_toy"]


def field_mean(sigma_dex):
    """Lognormal bias offset ``-sigma_s**2 / 2`` [natural-log units]."""
    return -0.5 * (jnp.asarray(sigma_dex) * jnp.log(10.0)) ** 2


@dataclass(frozen=True)
class ToyPopulation:
    """One synthetic population and the interim samples drawn from it.

    Parameters
    ----------
    fields : ndarray, shape (N, K, n)
        Centered-field draws [natural-log units].
    times_yr : ndarray, shape (n,)
        Time grid [yr].
    data : ndarray, shape (N, n)
        Observed data [natural-log units].
    noise_std : float
        White-noise standard deviation [natural-log units].
    sigma_true : float
        Injected amplitude [dex].
    tau_true_yr : float
        Injected timescale [yr].
    prior_sigma_bounds : tuple
        Prior support ``(low, high)`` for amplitude [dex].
    prior_tau_bounds_yr : tuple
        Prior support ``(low, high)`` for timescale [yr].
    """

    fields: jnp.ndarray  # (N, K, n) centered-field draws [natural-log units]
    times_yr: jnp.ndarray  # (n,) [yr]
    data: jnp.ndarray  # (N, n)
    noise_std: float
    sigma_true: float  # [dex]
    tau_true_yr: float  # [yr]
    prior_sigma_bounds: tuple
    prior_tau_bounds_yr: tuple


def _drw_cov(sigma_dex, tau_yr, times_yr):
    """Compute DRW covariance matrix.

    Parameters
    ----------
    sigma_dex : float
        Amplitude [dex].
    tau_yr : float
        Timescale [yr].
    times_yr : ndarray, shape (n,)
        Time grid [yr].

    Returns
    -------
    cov : ndarray, shape (n, n)
        Covariance matrix [natural-log units**2].
    """
    var = (sigma_dex * np.log(10.0)) ** 2
    dt = np.abs(times_yr[:, None] - times_yr[None, :])
    return var * np.exp(-dt / tau_yr)


def make_toy(
    *,
    n_galaxies,
    n_samples,
    n_grid,
    sigma_true,
    tau_true_yr,
    noise_std,
    prior_sigma_bounds,
    prior_tau_bounds_yr,
    seed,
):
    """Draw a population and exact interim posterior samples for each galaxy.

    Parameters
    ----------
    n_galaxies : int
        Population size.
    n_samples : int
        Interim draws per galaxy.
    n_grid : int
        Field grid points.
    sigma_true : float
        Injected amplitude [dex].
    tau_true_yr : float
        Injected timescale [yr].
    noise_std : float
        White-noise standard deviation [natural-log units].
    prior_sigma_bounds : tuple of float
        Interim prior support ``(low, high)`` for amplitude, uniform in
        ``sigma`` [dex].
    prior_tau_bounds_yr : tuple of float
        Interim prior support ``(low, high)`` for timescale, log-uniform in
        ``tau`` [yr].
    seed : int
        NumPy seed.

    Returns
    -------
    toy : ToyPopulation
    """
    rng = np.random.default_rng(seed)
    times = np.geomspace(1.0e6, 1.3e10, n_grid)
    cov = _drw_cov(sigma_true, tau_true_yr, times)
    mean = float(field_mean(sigma_true))

    truth = rng.multivariate_normal(np.full(n_grid, mean), cov, size=n_galaxies)
    data = truth + rng.normal(0.0, noise_std, size=truth.shape)

    # Exact interim posterior draws: sample (sigma, tau) from the per-galaxy
    # marginal on a fine grid, then the conditional field given (sigma, tau).
    # Interim sampling grid. Kept small on purpose: this fixture runs in the
    # fast test tier, and its cost is O(n_galaxies * n_interim**2) dense
    # factorizations in a Python loop.
    n_interim = 30
    g_sigma = np.linspace(*prior_sigma_bounds, n_interim)
    g_tau = np.geomspace(*prior_tau_bounds_yr, n_interim)
    fields = np.empty((n_galaxies, n_samples, n_grid))
    for i in range(n_galaxies):
        logw = np.empty((n_interim, n_interim))
        for a, s in enumerate(g_sigma):
            for b, t in enumerate(g_tau):
                k = _drw_cov(s, t, times) + noise_std**2 * np.eye(n_grid)
                r = data[i] - float(field_mean(s))
                _, logdet = np.linalg.slogdet(k)
                logw[a, b] = -0.5 * (r @ np.linalg.solve(k, r) + logdet)
        w = np.exp(logw - logw.max()).ravel()
        w /= w.sum()
        picks = rng.choice(w.size, size=n_samples, p=w)
        for k_idx, flat in enumerate(picks):
            s = g_sigma[flat // n_interim]
            t = g_tau[flat % n_interim]
            prior_cov = _drw_cov(s, t, times)
            post_cov = np.linalg.inv(np.linalg.inv(prior_cov) + np.eye(n_grid) / noise_std**2)
            post_mean = post_cov @ (
                np.linalg.solve(prior_cov, np.full(n_grid, float(field_mean(s))))
                + data[i] / noise_std**2
            )
            fields[i, k_idx] = rng.multivariate_normal(post_mean, post_cov)

    return ToyPopulation(
        fields=jnp.asarray(fields),
        times_yr=jnp.asarray(times),
        data=jnp.asarray(data),
        noise_std=float(noise_std),
        sigma_true=float(sigma_true),
        tau_true_yr=float(tau_true_yr),
        prior_sigma_bounds=tuple(prior_sigma_bounds),
        prior_tau_bounds_yr=tuple(prior_tau_bounds_yr),
    )


def closed_form_log_posterior(toy, grid_sigma, grid_tau_yr):
    """Analytic shared log-posterior on the flattened ``(sigma, tau)`` grid.

    The posterior is exact for this toy because the observation operator is
    the identity (field plus white noise), making the per-galaxy marginal
    likelihood a plain multivariate normal.

    Parameters
    ----------
    toy : ToyPopulation
    grid_sigma : array_like, shape (A,)
        Amplitude nodes [dex].
    grid_tau_yr : array_like, shape (B,)
        Timescale nodes [yr].

    Returns
    -------
    logp : ndarray, shape (A * B,)
        Unnormalized log-posterior [nats], C-ordered so that node ``a * B + b``
        is ``(grid_sigma[a], grid_tau_yr[b])``.
    """
    times = np.asarray(toy.times_yr)
    data = np.asarray(toy.data)
    n = times.size
    out = np.empty((len(grid_sigma), len(grid_tau_yr)))
    for a, s in enumerate(np.asarray(grid_sigma)):
        mean = np.full(n, float(field_mean(s)))
        for b, t in enumerate(np.asarray(grid_tau_yr)):
            k = _drw_cov(s, t, times) + toy.noise_std**2 * np.eye(n)
            _, logdet = np.linalg.slogdet(k)
            r = data - mean
            quad = np.einsum("ij,ij->i", r, np.linalg.solve(k, r.T).T)
            out[a, b] = float(np.sum(-0.5 * (quad + logdet)))
    return jnp.asarray(out.ravel())
