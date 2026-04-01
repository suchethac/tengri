# Copyright 2020- The Blackjax Authors.
# Ported to tengri from github.com/handley-lab/blackjax (nested_sampling branch).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for Nested Sampling post-processing.

Ported from blackjax.ns.utils.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from tengri.inference.ns._types import Array, ArrayTree, PRNGKey
from tengri.inference.ns.base import NSInfo, NSState

__all__ = [
    "compute_num_live",
    "ess",
    "finalise",
    "log1mexp",
    "logX",
    "log_weights",
    "sample",
    "uniform_prior",
]


def log1mexp(x: Array) -> Array:
    """Compute log(1 - exp(x)) in a numerically stable way."""
    return jnp.where(
        x > -0.6931472,  # approx log(2)
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_num_live(info: NSInfo) -> Array:
    """Compute the effective number of live points at each death contour.

    When doing batch deletions, the jump in energy level can be smoothed by
    transforming 1 jump of size k into k jumps of size 1.

    Returns
    -------
    Array
        Effective number of live points for each dead particle.
    """
    birth_logL = info.particles.loglikelihood_birth
    death_logL = info.particles.loglikelihood

    birth_events = jnp.column_stack((birth_logL, jnp.ones_like(birth_logL, dtype=int)))
    death_events = jnp.column_stack((death_logL, -jnp.ones_like(death_logL, dtype=int)))
    combined = jnp.concatenate([birth_events, death_events], axis=0)
    logL_col = combined[:, 0]
    n_col = combined[:, 1]
    not_nan_sort_key = ~jnp.isnan(logL_col)
    logL_sort_key = logL_col
    n_sort_key = n_col
    sorted_indices = jnp.lexsort((n_sort_key, logL_sort_key, not_nan_sort_key))
    sorted_n_col = n_col[sorted_indices]
    cumsum = jnp.cumsum(sorted_n_col)
    cumsum = jnp.maximum(cumsum, 0)
    death_mask_sorted = sorted_n_col == -1
    num_live = cumsum[death_mask_sorted] + 1
    return num_live


def logX(rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100) -> tuple[Array, Array]:
    """Simulate the stochastic evolution of log prior volumes.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    dead_info
        An NSInfo object containing loglikelihood_birth and loglikelihood
        for all dead particles (assumed sorted by death log-likelihood).
    shape
        Number of Monte Carlo samples for stochastic log-volume sequences.

    Returns
    -------
    tuple[Array, Array]
        - logX_cumulative: shape (n_dead, shape)
        - log_dX_elements: shape (n_dead, shape)
    """
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(
        subkey,
        shape=(dead_info.particles.loglikelihood.shape[0], shape),
    )
    r = jax.lax.log1p(jax.lax.neg(u))
    num_live = compute_num_live(dead_info)
    t = r / num_live[:, jnp.newaxis]
    logX_vals = jnp.cumsum(t, axis=0)

    logXp = jnp.concatenate([jnp.zeros((1, logX_vals.shape[1])), logX_vals[:-1]], axis=0)
    logXm = jnp.concatenate([logX_vals[1:], jnp.full((1, logX_vals.shape[1]), -jnp.inf)], axis=0)
    log_diff = logXm - logXp
    logdX = log1mexp(log_diff) + logXp - jnp.log(2)
    return logX_vals, logdX


def log_weights(rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100, beta: float = 1.0) -> Array:
    """Calculate log importance weights for Nested Sampling results.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    dead_info
        NSInfo containing loglikelihood_birth and loglikelihood for all dead particles.
    shape
        Number of MC samples for simulating log(dX_i).
    beta
        Inverse temperature. Typically 1.0.

    Returns
    -------
    Array
        Log importance weights, shape (n_dead, shape).
    """
    sort_indices = jnp.argsort(dead_info.particles.loglikelihood)
    unsort_indices = jnp.empty_like(sort_indices)
    unsort_indices = unsort_indices.at[sort_indices].set(jnp.arange(len(sort_indices)))
    dead_info_sorted = jax.tree.map(lambda x: x[sort_indices], dead_info)
    _, log_dX = logX(rng_key, dead_info_sorted, shape)
    log_w = log_dX + beta * dead_info_sorted.particles.loglikelihood[..., jnp.newaxis]
    return log_w[unsort_indices]


def finalise(live: NSState, dead: list[NSInfo], update_info: bool = True) -> NSInfo:
    """Combine dead particle history with final live points.

    Parameters
    ----------
    live
        Final NSState containing live particles.
    dead
        List of NSInfo objects from each NS step.
    update_info
        Whether to concatenate update_info from dead list.

    Returns
    -------
    NSInfo
        Single NSInfo with all particles concatenated.
    """
    if update_info:
        update_infos = [d.update_info for d in dead]
        final_update_info = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *update_infos
        )
    else:
        final_update_info = None

    particles = [d.particles for d in dead] + [live.particles]
    final_particles = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *particles)
    return NSInfo(final_particles, final_update_info)


def ess(rng_key: PRNGKey, dead: NSInfo) -> Array:
    """Compute effective sample size from log-weights.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    dead
        NSInfo with all dead (and final live) particles (from ``finalise``).

    Returns
    -------
    Array
        Mean effective sample size (scalar).
    """
    logw = log_weights(rng_key, dead).mean(axis=-1)
    logw -= logw.max()
    l_sum_w = jax.scipy.special.logsumexp(logw)
    l_sum_w_sq = jax.scipy.special.logsumexp(2 * logw)
    return jnp.exp(2 * l_sum_w - l_sum_w_sq)


def sample(rng_key: PRNGKey, dead: NSInfo, shape: int = 1000) -> ArrayTree:
    """Resample particles according to importance weights.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    dead
        NSInfo with all particles.
    shape
        Number of posterior samples.

    Returns
    -------
    ArrayTree
        Resampled particles.
    """
    logw = log_weights(rng_key, dead).mean(axis=-1)
    indices = jax.random.choice(
        rng_key,
        dead.particles.loglikelihood.shape[0],
        p=jnp.exp(logw.squeeze() - jnp.max(logw)),
        shape=(shape,),
        replace=True,
    )
    return jax.tree.map(lambda leaf: leaf[indices], dead.particles)


def get_first_row(x: ArrayTree) -> ArrayTree:
    """Extract the first element along the leading axis of each leaf."""
    return jax.tree.map(lambda x: x[0], x)


def uniform_prior(
    rng_key: PRNGKey, num_live: int, bounds: dict[str, tuple[float, float]]
) -> tuple[ArrayTree, Callable]:
    """Create a uniform prior for parameters.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    num_live
        Number of live particles to sample.
    bounds
        Parameter name → (min, max) bounds.

    Returns
    -------
    tuple
        (particles, logprior_fn).
    """

    def logprior_fn(params):
        logprior = 0.0
        for p, (a, b) in bounds.items():
            x = params[p]
            logprior += jax.scipy.stats.uniform.logpdf(x, a, b - a)
        return logprior

    def prior_sample(rng_key):
        init_keys = jax.random.split(rng_key, len(bounds))
        params = {}
        for rng_key, (p, (a, b)) in zip(init_keys, bounds.items()):
            params[p] = jax.random.uniform(rng_key, minval=a, maxval=b)
        return params

    init_keys = jax.random.split(rng_key, num_live)
    particles = jax.vmap(prior_sample)(init_keys)

    return particles, logprior_fn
