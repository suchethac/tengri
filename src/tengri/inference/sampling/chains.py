# SPDX-License-Identifier: BSD-3-Clause
"""The multi-chain layer: independent overdispersed starts from one adaptation."""

from __future__ import annotations

import jax


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
    chain_jitter : float
        Standard deviation of each offset.

    Returns
    -------
    ndarray, shape (n_chains, D)
        ``base + chain_jitter * z`` with ``z`` standard normal, one row per chain.
    """
    chain_init_key = jax.random.fold_in(warmup_key, 0xC4A15)
    return base[None, :] + chain_jitter * jax.random.normal(
        chain_init_key, (n_chains, base.shape[0]), dtype=base.dtype
    )


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
