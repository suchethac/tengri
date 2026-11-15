# SPDX-License-Identifier: BSD-3-Clause
"""The multi-chain layer: independent overdispersed starts from one adaptation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: Start dispersion of the extra chains in units of the adapted posterior standard
#: deviation. Starts that are narrower than the posterior make a cross-chain R-hat a
#: consistency check; two sd apart they are a test. Used where the adaptation provides a
#: metric (its inverse mass matrix estimates the posterior covariance).
OVERDISPERSION = 2.0

#: Start dispersion where no posterior scale is known: the sd of the standardized prior the
#: flat space is built on (iid N(0, 1)), which the posterior only narrows.
PRIOR_SCALE = 1.0

#: Start dispersion of a family with no adapted metric (Barker, MALA) in the raw coordinates,
#: where no posterior scale is known: a step takes it a few posterior widths at best, so a
#: prior-scale start on a sharp posterior never arrives. In whitened coordinates (the dense
#: metric layer) the posterior has unit scale and ``OVERDISPERSION`` applies instead.
UNADAPTED_START_SCALE = 0.1

#: Cap of the isotropic part of the start dispersion in adapted posterior standard
#: deviations: a start farther than this from a sharp posterior (a prior-scale start of 250
#: widths) is a transient no short chain recovers from, which R-hat then reports as
#: non-convergence rather than as the multimodality it was meant to find.
START_CAP = 10.0


def jittered_starts(base, warmup_key, n_chains, chain_jitter):
    """Return ``n_chains`` independent overdispersed starts around ``base``.

    Parameters
    ----------
    base : ndarray, shape (D,)
        The position every chain is offset from.
    warmup_key : PRNGKey
        Key of the adaptation; the offsets are folded off it, independent of the chains'
        own sampling keys.
    n_chains : int (static)
        Number of starts.
    chain_jitter : float or ndarray
        Standard deviation of each offset: a scalar, a per-coordinate vector, shape (D,), or
        a lower-triangular factor, shape (D, D), of the offset covariance.

    Returns
    -------
    ndarray, shape (n_chains, D)
        ``base + chain_jitter * z`` (``z`` standard normal, one row per chain), with the
        vector or factor applied coordinate-wise or as a matrix.
    """
    chain_init_key = jax.random.fold_in(warmup_key, 0xC4A15)
    z = jax.random.normal(chain_init_key, (n_chains, base.shape[0]), dtype=base.dtype)
    scale = jnp.asarray(chain_jitter)
    if scale.ndim == 2:
        return base[None, :] + z @ scale.T
    return base[None, :] + scale * z


def start_scale(inverse_mass_matrix, chain_jitter, factor=OVERDISPERSION):
    """Return the start dispersion of the extra chains.

    The start covariance is ``factor^2 Sigma + diag(s^2)`` with ``Sigma`` the adaptation's
    estimate of the posterior covariance and ``s = min(chain_jitter, START_CAP sd)`` per
    coordinate: at least ``factor`` posterior standard deviations everywhere, and up to the
    prior's scale (``chain_jitter``) where the posterior is not much narrower, so that two
    modes a few widths apart but inside the prior are both visited.

    Parameters
    ----------
    inverse_mass_matrix : ndarray, shape (D,) or (D, D), or None
        The adaptation's estimate of the posterior covariance, diagonal or dense.
    chain_jitter : float
        Isotropic scale: the whole dispersion when the adaptation has no metric.
    factor : float
        Multiple of the posterior standard deviation.

    Returns
    -------
    float or ndarray
        ``chain_jitter`` without a metric; else the per-coordinate sd, shape (D,), or the
        lower Cholesky factor, shape (D, D), of the start covariance.
    """
    if inverse_mass_matrix is None or not hasattr(inverse_mass_matrix, "ndim"):
        return chain_jitter
    if inverse_mass_matrix.ndim == 1:
        iso = jnp.minimum(chain_jitter, START_CAP * jnp.sqrt(inverse_mass_matrix))
        return jnp.sqrt(factor**2 * inverse_mass_matrix + iso**2)
    sd = jnp.sqrt(jnp.diag(inverse_mass_matrix))
    iso = jnp.minimum(chain_jitter, START_CAP * sd)
    return jnp.linalg.cholesky(factor**2 * inverse_mass_matrix + jnp.diag(iso**2))


def run_jittered_chains(sample_body, init_state, starts, chain_keys):
    """Sample one chain per start with one adapted kernel.

    Parameters
    ----------
    sample_body : callable
        ``sample_body(state, keys) -> Draw`` for one chain.
    init_state : callable
        ``init_state(position) -> state`` of one chain.
    starts : ndarray, shape (n_chains, D)
        Chain start positions.
    chain_keys : ndarray, shape (n_chains, n_iter, 2)
        One key stream per chain.

    Returns
    -------
    Draw
        Sampling outputs with a leading ``n_chains`` axis.
    """
    states = jax.vmap(init_state)(starts)
    return jax.vmap(sample_body)(states, chain_keys)
