# SPDX-License-Identifier: BSD-3-Clause
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
"""Core types and base kernel for Nested Sampling.

Implements the same algorithms as blackjax.ns.base.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.inference.backends.nested._types import Array, ArrayLikeTree, PRNGKey

__all__ = ["NSInfo", "NSState", "build_kernel", "delete_fn", "init"]


class StateWithLogLikelihood(NamedTuple):
    """State of a particle in NS.

    Attributes
    ----------
    position
        The position of the particle (PyTree).
    logdensity
        The log-density under the prior.
    loglikelihood
        The log-likelihood.
    loglikelihood_birth
        The log-likelihood birth threshold.
    """

    position: ArrayLikeTree
    logdensity: Array
    loglikelihood: Array
    loglikelihood_birth: Array


class NSState(NamedTuple):
    """State of the Nested Sampler."""

    particles: StateWithLogLikelihood


class NSInfo(NamedTuple):
    """Information returned at each NS step.

    Attributes
    ----------
    particles
        The StateWithLogLikelihood of dead particles.
    update_info
        Information from the inner kernel update step.
    """

    particles: StateWithLogLikelihood
    update_info: NamedTuple


def init_state_strategy(
    position: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: float = jnp.nan,
    data_args: dict | None = None,
) -> StateWithLogLikelihood:
    """Default initialization strategy for each state.

    Parameters
    ----------
    position
        A PyTree of arrays representing the particle position.
    logprior_fn
        Log-prior density function for a single particle.
    loglikelihood_fn
        Log-likelihood function for a single particle.
        If *data_args* is provided, called as ``loglikelihood_fn(position, data_args)``
        (2-arg, compile-once mode).  Otherwise ``loglikelihood_fn(position)`` (1-arg).
    loglikelihood_birth
        Log-likelihood threshold the particle must exceed.
    data_args : dict, optional
        Observed-data dict passed as a traced JAX argument for
        compile-once reuse across galaxies.

    Returns
    -------
    StateWithLogLikelihood
    """
    logprior_values = logprior_fn(position)
    if data_args is not None:
        loglikelihood_values = loglikelihood_fn(position, data_args)
    else:
        loglikelihood_values = loglikelihood_fn(position)
    loglikelihood_birth_values = loglikelihood_birth * jnp.ones_like(loglikelihood_values)

    return StateWithLogLikelihood(
        position, logprior_values, loglikelihood_values, loglikelihood_birth_values
    )


def init(
    positions: ArrayLikeTree,
    init_state_fn: Callable,
    loglikelihood_birth: float = jnp.nan,
) -> NSState:
    """Initialize the Nested Sampler state.

    Parameters
    ----------
    positions
        Initial positions drawn from the prior (leading dim = n_particles).
    init_state_fn
        Function that initializes state from positions.
    loglikelihood_birth
        Initial log-likelihood birth threshold.

    Returns
    -------
    NSState
    """
    state_init = init_state_fn(positions)
    loglikelihood_birth_array = loglikelihood_birth * jnp.ones_like(state_init.loglikelihood_birth)
    return NSState(state_init._replace(loglikelihood_birth=loglikelihood_birth_array))


def build_kernel(
    delete_fn: Callable,
    inner_kernel: Callable,
) -> Callable:
    """Build a generic Nested Sampling kernel.

    Parameters
    ----------
    delete_fn
        ``(state) -> (dead_idx, target_update_idx)``.
    inner_kernel
        ``(rng_key, state, loglikelihood_0) -> (new_particles, info)``.

    Returns
    -------
    Callable
        ``(rng_key, state) -> (new_state, ns_info)``.
    """

    def kernel(rng_key: PRNGKey, state: NSState) -> tuple[NSState, NSInfo]:
        """Execute one NS step: delete worst particle, propose replacement, record dead."""
        # Delete and grab dead information
        dead_idx, target_update_idx = delete_fn(state)
        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)

        # Generate replacement particles
        rng_key, inner_key = jax.random.split(rng_key)
        loglikelihood_0 = dead_particles.loglikelihood.max()
        new_particles, inner_update_info = inner_kernel(inner_key, state, loglikelihood_0)

        # Update the particles
        state = state._replace(
            particles=jax.tree_util.tree_map(
                lambda p, n: p.at[target_update_idx].set(n),
                state.particles,
                new_particles,
            )
        )

        info = NSInfo(dead_particles, inner_update_info)
        return state, info

    return kernel


def delete_fn(state: NSState, num_delete: int) -> tuple[Array, Array]:
    """Identify particles to delete (lowest log-likelihoods).

    Parameters
    ----------
    state
        Current NS state.
    num_delete
        Number of particles to delete.

    Returns
    -------
    tuple[Array, Array]
        (dead_idx, target_update_idx).
    """
    loglikelihood = state.particles.loglikelihood
    _, dead_idx = jax.lax.top_k(-loglikelihood, num_delete)
    target_update_idx = dead_idx
    return dead_idx, target_update_idx
