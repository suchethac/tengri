# SPDX-License-Identifier: BSD-3-Clause
"""Two-step hierarchical estimators for the shared SFH PSD block."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tengri.inference.population.kernel import ou_logpdf

__all__ = ["SharedGrid", "effective_sample_size", "shared_log_posterior"]


def _field_mean(sigma_dex):
    return -0.5 * (jnp.asarray(sigma_dex) * jnp.log(10.0)) ** 2


@dataclass(frozen=True)
class SharedGrid:
    """Quadrature grid over the shared ``(sigma, tau)`` block.

    Attributes
    ----------
    sigma : ndarray, shape (A,)
        Amplitude nodes [dex].
    tau_yr : ndarray, shape (B,)
        Timescale nodes [yr].
    log_prior : ndarray, shape (A * B,)
        Log prior density at each node [nats], C-ordered so node ``a * B + b``
        is ``(sigma[a], tau_yr[b])``.
    log_volume : ndarray, shape (A * B,)
        Log quadrature weight of each node [nats].
    """

    sigma: jnp.ndarray
    tau_yr: jnp.ndarray
    log_prior: jnp.ndarray
    log_volume: jnp.ndarray

    @classmethod
    def uniform(cls, *, sigma_bounds, tau_bounds_yr, n_sigma, n_tau):
        """Grid uniform in ``sigma`` and log-uniform in ``tau``.

        Parameters
        ----------
        sigma_bounds : tuple of float
            ``(lo, hi)`` amplitude support [dex].
        tau_bounds_yr : tuple of float
            ``(lo, hi)`` timescale support [yr].
        n_sigma, n_tau : int
            Node counts.

        Returns
        -------
        grid : SharedGrid
        """
        sigma = jnp.linspace(sigma_bounds[0], sigma_bounds[1], n_sigma)
        tau = jnp.geomspace(tau_bounds_yr[0], tau_bounds_yr[1], n_tau)
        d_sigma = (sigma_bounds[1] - sigma_bounds[0]) / n_sigma
        d_log_tau = (jnp.log(tau_bounds_yr[1]) - jnp.log(tau_bounds_yr[0])) / n_tau
        n_nodes = n_sigma * n_tau
        log_prior = jnp.full((n_nodes,), -jnp.log(n_nodes))
        log_volume = jnp.full((n_nodes,), jnp.log(d_sigma) + jnp.log(d_log_tau))
        return cls(sigma=sigma, tau_yr=tau, log_prior=log_prior, log_volume=log_volume)

    @property
    def nodes(self):
        """``(A * B, 2)`` array of ``(sigma [dex], tau [yr])`` pairs."""
        a = jnp.repeat(self.sigma, self.tau_yr.size)
        b = jnp.tile(self.tau_yr, self.sigma.size)
        return jnp.stack([a, b], axis=-1)


def effective_sample_size(log_weights):
    r"""Kish effective sample size of log-domain importance weights.

    .. math:: \mathrm{ESS} = \frac{\left(\sum_k w_k\right)^2}{\sum_k w_k^2}

    Parameters
    ----------
    log_weights : array_like, shape (..., K)
        Unnormalized log weights [nats]; the reduction is over the last axis.

    Returns
    -------
    ess : ndarray, shape (...)
        Effective number of draws [dimensionless], in ``[1, K]``.
    """
    lw = jnp.asarray(log_weights)
    lw = lw - jnp.max(lw, axis=-1, keepdims=True)
    w = jnp.exp(lw)
    return jnp.sum(w, axis=-1) ** 2 / jnp.sum(w**2, axis=-1)


def _node_logpdf_table(fields, times_yr, nodes):
    """``(n_nodes, N, K)`` field log-density at every grid node."""

    def one_node(node):
        sigma, tau = node[0], node[1]
        mean = _field_mean(sigma)
        per_sample = jax.vmap(lambda m: ou_logpdf(m, mean, sigma, tau, times_yr))
        return jax.vmap(per_sample)(fields)

    return jax.lax.map(one_node, nodes)


def shared_log_posterior(fields, times_yr, grid, *, method="b2"):
    r"""Shared ``(sigma, tau)`` log-posterior from per-galaxy interim samples.

    The population factorizes given the shared block, so

    .. math::

        \log p(\sigma, \tau \mid \{d\}) = \log p(\sigma, \tau)
          + \sum_{i=1}^{N} \log\!\left[\frac{1}{K}\sum_{k}
            \frac{\mathcal{N}\big(m_i^{(k)};\ \mu(\sigma),\ K(\sigma,\tau)\big)}
                 {p_0\big(m_i^{(k)}\big)}\right]

    where :math:`m_i^{(k)}` are interim posterior draws of the centered field
    [natural-log units] and :math:`p_0` is the interim pushforward prior,
    evaluated by the same quadrature as the numerator so one grid serves both.

    Parameters
    ----------
    fields : array_like, shape (N, K, n)
        Interim centered-field draws [natural-log units].
    times_yr : array_like, shape (n,)
        Physical times [yr].
    grid : SharedGrid
        Quadrature grid.
    method : {"b2", "b1"}, optional
        ``"b2"`` (default) is the reweighting estimator. ``"b1"`` is the
        marginal-posterior product, retained as an independent cross-check
        whose error mode is different; see Notes.

    Returns
    -------
    log_posterior : ndarray, shape (A * B,)
        Unnormalized log-posterior [nats] on ``grid.nodes``.
    ess : ndarray, shape (N,)
        Per-galaxy effective sample size at the posterior mode [dimensionless].

    Notes
    -----
    **JIT/vmap compatible**: yes. Cost is ``O(A B N K n)`` with no matrix
    factorization, because :func:`ou_logpdf` exploits the Markov structure.

    ``"b2"`` fails by importance-weight degeneracy, which the returned ``ess``
    measures directly. ``"b1"`` fails by compounding density-estimation bias in
    the tails, which is *not* observable from inside the estimator -- multiplying
    N kernel density estimates whose tails err in a common direction shifts the
    result without widening it. Prefer ``"b2"``; use ``"b1"`` only to disagree
    with it.
    """
    fields = jnp.asarray(fields)
    times_yr = jnp.asarray(times_yr)
    if method not in ("b1", "b2"):
        raise ValueError(
            f"method must be 'b2' (production) or 'b1' (cross-check), got {method!r}."
        )

    table = _node_logpdf_table(fields, times_yr, grid.nodes)  # (G, N, K)

    # Interim pushforward prior p_0(m), same quadrature as the numerator.
    log_p0 = jax.scipy.special.logsumexp(
        table + (grid.log_prior + grid.log_volume)[:, None, None], axis=0
    )  # (N, K)

    log_w = table - log_p0[None, :, :]  # (G, N, K)
    per_galaxy = jax.scipy.special.logsumexp(log_w, axis=-1) - jnp.log(fields.shape[1])  # (G, N)

    if method == "b1":
        # Marginal-posterior product: drop the pushforward correction and use
        # the interim marginal directly. Deliberately a different estimator.
        per_galaxy = jax.scipy.special.logsumexp(table, axis=-1) - jnp.log(fields.shape[1])

    log_posterior = grid.log_prior + jnp.sum(per_galaxy, axis=-1)
    best = jnp.argmax(log_posterior)
    ess = effective_sample_size(log_w[best])
    return log_posterior, ess
