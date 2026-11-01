# SPDX-License-Identifier: BSD-3-Clause
"""Importance sampling shared by the inference backends.

One Student-t importance sampler serves the Laplace backend (its importance
check) and the ``hmc_is`` backend (its evidence estimate). The target is always
the flat log-density the samplers use, so the two routes score the same posterior.

Pipeline: :func:`sample_student_t` draws from the proposal,
:func:`evaluate_logdensity` scores the draws with the target in batches, and
:func:`importance_estimate` forms the weights, the evidence estimate and its
delta-method error, and the PSIS smoothing from ``tengri.inference.backends._psis``.
:func:`importance_sample` runs the three steps.

The evaluation is a ``jax.lax.map`` over batches, not a full ``vmap``: a full vmap
over thousands of draws is a known memory trap in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from tengri.inference.backends._psis import PSISResult, psis


@dataclass(frozen=True)
class StudentTProposal:
    """Student-t proposal for importance sampling.

    Attributes
    ----------
    mean : jnp.ndarray
        Mean of the proposal, shape (D,).
    chol : jnp.ndarray
        Lower Cholesky factor of the proposal covariance, shape (D, D).
        The full covariance is cov = chol @ chol.T.
    df : float
        Degrees of freedom of the Student-t distribution.
    """

    mean: jnp.ndarray
    chol: jnp.ndarray
    df: float


@dataclass(frozen=True)
class ImportanceResult:
    """Output of :func:`importance_sample` and :func:`importance_estimate`.

    Attributes
    ----------
    draws : jnp.ndarray
        Proposal draws, shape ``(n, D)``.
    log_ratios : numpy.ndarray
        Raw log importance ratios ``logdensity + log_norm - log_q``, shape ``(n,)``.
    log_z : float
        Raw importance-sampling estimate of the log normalizing constant,
        ``logsumexp(log_ratios) - log(n)``.
    log_z_err : float
        Delta-method standard error of ``log_z`` from the raw weights.
    ess : float
        Effective sample size ``(sum w)^2 / sum w^2`` of the raw weights, an
        absolute count.
    max_weight_frac : float
        Largest raw weight as a fraction of the total weight.
    psis : PSISResult
        Pareto-smoothed log weights and the tail shape ``pareto_k``.
    """

    draws: jnp.ndarray
    log_ratios: np.ndarray
    log_z: float
    log_z_err: float
    ess: float
    max_weight_frac: float
    psis: PSISResult


def student_t_logpdf(proposal: StudentTProposal, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate Student-t log-probability density.

    Computes log p(x; μ, Σ, df) where Σ = chol @ chol.T.

    Parameters
    ----------
    proposal : StudentTProposal
        Proposal object with mean, Cholesky factor, and degrees of freedom.
    x : jnp.ndarray
        Points at which to evaluate the density, shape (n, D) or (D,).

    Returns
    -------
    jnp.ndarray
        Log probability density, shape (n,) or scalar.

    Notes
    -----
    Uses the Student-t density formula:
    log p(x; μ, Σ, ν) = log Γ((ν+D)/2) - log Γ(ν/2) - (D/2) log(πν)
                        - (1/2) log det(Σ)
                        - ((ν+D)/2) log(1 + (x-μ)^T Σ^-1 (x-μ) / ν)

    This matches scipy.stats.multivariate_t with shape=cov (the covariance).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = proposal.mean
    chol = proposal.chol
    nu = proposal.df
    dim = len(mu)

    # Standardize: (x - mu)^T Σ^-1 (x - mu) = ||L^-1 @ (x - mu)||^2 for Σ = L L^T.
    if x.ndim == 1:
        delta = x - mu
    else:
        delta = x - mu[None, :]

    # Solve L @ y = delta^T (shape (..., D))
    y = jnp.linalg.solve(chol, delta.T).T  # Shape (..., D) or (D,)
    mahal_sq = jnp.sum(y**2, axis=-1)  # Shape (...,) or scalar

    # log det(Σ) = 2 * sum(log diag(L))
    log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))

    const = (
        float(special.loggamma((nu + dim) / 2.0))
        - float(special.loggamma(nu / 2.0))
        - (dim / 2.0) * jnp.log(jnp.pi * nu)
    )

    return const - 0.5 * log_det_sigma - ((nu + dim) / 2.0) * jnp.log(1.0 + mahal_sq / nu)


def sample_student_t(key: jax.Array, proposal: StudentTProposal, n_draws: int) -> jnp.ndarray:
    """Draw ``n_draws`` points from the Student-t proposal.

    Uses ``x = mu + L @ z / sqrt(g / nu)`` with ``z ~ N(0, I)`` and
    ``g ~ chi^2(nu)``, consuming two splits of ``key`` in that order.

    Returns
    -------
    jnp.ndarray
        Draws, shape ``(n_draws, D)``.
    """
    mu = proposal.mean
    chol = proposal.chol
    nu = proposal.df
    dim = len(mu)

    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (n_draws, dim))

    key, subkey = jax.random.split(key)
    g = jax.random.chisquare(subkey, nu, (n_draws,))

    scale = jnp.sqrt(g / nu)
    z_scaled = z / scale[:, None]
    return mu[None, :] + z_scaled @ chol.T


@partial(jax.jit, static_argnums=(0, 3))
def _evaluate_logdensity_jit(logdensity_fn, positions, data_args, batch_size):
    """Map ``logdensity_fn`` over rows of ``positions`` in batches of ``batch_size``."""
    return jax.lax.map(lambda x: logdensity_fn(x, data_args), positions, batch_size=batch_size)


def evaluate_logdensity(logdensity_fn, positions, data_args, batch_size: int = 256) -> jnp.ndarray:
    """Evaluate a flat log-density at each row of ``positions``.

    Parameters
    ----------
    logdensity_fn : callable
        ``logdensity_fn(position, data_args) -> scalar``, JAX-traceable. Static
        under :func:`jax.jit`, so the compiled program is cached per function.
    positions : array_like, shape (n, D)
        Points to score.
    data_args : pytree
        Passed through to ``logdensity_fn``; traced, not closed over.
    batch_size : int, default 256
        Rows evaluated per batch. Bounds the memory of the evaluation.

    Returns
    -------
    jnp.ndarray
        Log-density at each point, shape ``(n,)``.
    """
    positions = jnp.asarray(positions, dtype=jnp.float64)
    return _evaluate_logdensity_jit(logdensity_fn, positions, data_args, int(batch_size))


def importance_estimate(
    draws: jnp.ndarray,
    log_target: np.ndarray,
    log_q: np.ndarray,
    log_norm: float = 0.0,
) -> ImportanceResult:
    """Form importance weights, the evidence estimate, and PSIS from scored draws.

    Parameters
    ----------
    draws : jnp.ndarray, shape (n, D)
        Proposal draws the scores refer to.
    log_target : array_like, shape (n,)
        Unnormalized log-target at each draw.
    log_q : array_like, shape (n,)
        Log proposal density at each draw.
    log_norm : float, default 0.0
        Constant added to ``log_target``, for example ``-(D/2) log 2 pi`` to
        normalize an unnormalized standard-normal prior.

    Returns
    -------
    ImportanceResult

    Raises
    ------
    ValueError
        If the log ratios contain NaN or ``+inf``, or fewer than 10 entries.
    """
    log_ratios = (
        np.asarray(log_target, dtype=float) + float(log_norm) - np.asarray(log_q, dtype=float)
    )
    smoothed = psis(log_ratios)

    n_draws = log_ratios.size
    weights = np.exp(log_ratios - np.max(log_ratios))
    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights**2))

    mean_w = sum_w / n_draws
    var_w = sum_w2 / n_draws - mean_w**2
    std_w = float(np.sqrt(max(var_w, 1e-16)))

    return ImportanceResult(
        draws=draws,
        log_ratios=log_ratios,
        log_z=float(smoothed.log_mean_raw_weight),
        log_z_err=float(std_w / (mean_w * np.sqrt(n_draws))),
        ess=float(sum_w**2 / sum_w2),
        max_weight_frac=float(np.max(weights) / sum_w),
        psis=smoothed,
    )


def importance_sample(
    logdensity_fn,
    data_args,
    proposal: StudentTProposal,
    n_draws: int,
    key: jax.Array,
    *,
    log_norm: float = 0.0,
    batch_size: int = 256,
) -> ImportanceResult:
    """Importance-sample the target ``exp(logdensity_fn + log_norm)`` with a Student-t proposal.

    Parameters
    ----------
    logdensity_fn : callable
        Flat log-density, ``(position, data_args) -> scalar``, JAX-traceable.
    data_args : pytree
        Passed through to ``logdensity_fn``.
    proposal : StudentTProposal
        Proposal distribution.
    n_draws : int
        Number of proposal draws, at least 10.
    key : jax.Array
        PRNG key for the proposal draws.
    log_norm : float, default 0.0
        Constant added to the log-density before weighting.
    batch_size : int, default 256
        Rows of the target evaluated per batch.

    Returns
    -------
    ImportanceResult
    """
    draws = sample_student_t(key, proposal, n_draws)
    log_target = evaluate_logdensity(logdensity_fn, draws, data_args, batch_size=batch_size)
    log_q = student_t_logpdf(proposal, draws)
    return importance_estimate(draws, np.asarray(log_target), np.asarray(log_q), log_norm)
