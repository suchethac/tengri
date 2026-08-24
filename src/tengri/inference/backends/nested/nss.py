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
"""Nested Slice Sampling (NSS) algorithm.

A specific implementation of Nested Sampling that uses Hit-and-Run Slice
Sampling (HRSS) as the inner MCMC kernel.

Implements the same algorithms as blackjax.ns.nss.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from tengri.inference.backends.nested._types import ArrayTree, SamplingAlgorithm
from tengri.inference.backends.nested.adaptive import build_kernel as build_adaptive_kernel, init
from tengri.inference.backends.nested.base import (
    NSInfo,
    NSState,
    delete_fn as default_delete_fn,
    init_state_strategy,
)
from tengri.inference.backends.nested.from_mcmc import update_with_mcmc_take_last
from tengri.inference.backends.nested.particles import particles_covariance_matrix
from tengri.inference.backends.nested.slice_sampling import (
    build_kernel as build_slice_kernel,
    sample_direction_from_covariance,
)

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
    "update_inner_kernel_params",
]


def default_stepper_fn(x: ArrayTree, d: ArrayTree, t: float) -> tuple[ArrayTree, bool]:
    """Simple stepper: ``x_new = x + t * d``.

    Parameters
    ----------
    x
        Starting position (PyTree).
    d
        Direction of movement (PyTree, same structure as x).
    t
        Scalar step size along the direction.

    Returns
    -------
    tuple[ArrayTree, bool]
        (new position, True).
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d), True


def update_inner_kernel_params(
    rng_key: jax.random.PRNGKey,
    state: NSState,
    info: NSInfo,
    inner_kernel_params: dict[str, ArrayTree] | None = None,
) -> dict[str, ArrayTree]:
    """Update inner kernel parameters from current particles.

    Computes the empirical covariance matrix from the live particles.
    Preserves ``data_args`` from previous params when present (compile-once mode).

    Parameters
    ----------
    rng_key
        PRNG key (unused but kept for interface consistency).
    state
        Current NSState.
    info
        Information from the last NS step (unused).
    inner_kernel_params
        Previous parameters; ``data_args`` key is preserved if present.

    Returns
    -------
    dict
        Dictionary with updated 'cov' (and preserved 'data_args' if applicable).
    """
    result = {"cov": jnp.atleast_2d(particles_covariance_matrix(state.particles.position))}
    if inner_kernel_params is not None and "data_args" in inner_kernel_params:
        result["data_args"] = inner_kernel_params["data_args"]
    return result


def build_kernel(
    init_state_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build the Nested Slice Sampling kernel.

    See ``as_top_level_api`` for parameter descriptions.
    """

    def constrained_mcmc_slice_fn(rng_key, state, loglikelihood_0, **params):
        """Single slice sampling step respecting the NS likelihood constraint."""
        data_args = params.pop("data_args", None)
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = generate_slice_direction_fn(prop_key, state.position, **params)

        def slice_fn(t) -> tuple[NSState, bool]:
            """Evaluate state at parameter t along the slice direction."""
            x, step_accepted = stepper_fn(state.position, d, t)
            new_state = init_state_fn(x, loglikelihood_birth=loglikelihood_0, data_args=data_args)
            in_contour = new_state.loglikelihood > loglikelihood_0
            is_accepted = in_contour & step_accepted
            return new_state, is_accepted

        slice_kernel = build_slice_kernel(
            slice_fn,
            max_steps=max_steps,
            max_shrinkage=max_shrinkage,
        )
        new_slice_state, slice_info = slice_kernel(rng_key, state)
        return new_slice_state, slice_info

    inner_kernel = update_strategy(constrained_mcmc_slice_fn, num_inner_steps, num_delete)

    delete_fn = partial(delete_fn, num_delete=num_delete)

    kernel = build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    init_state_strategy_fn: Callable = init_state_strategy,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    max_steps: int = 10,
    max_shrinkage: int = 100,
    data_args: dict | None = None,
) -> SamplingAlgorithm:
    """Create a Nested Slice Sampling (NSS) algorithm.

    Configures Nested Sampling with HRSS as the inner kernel. The HRSS
    direction proposal covariance is managed via
    ``update_inner_kernel_params``.

    Parameters
    ----------
    logprior_fn
        Log-prior probability of a single particle.
    loglikelihood_fn
        Log-likelihood of a single particle.
        If *data_args* is provided, must be 2-arg: ``(params, data_args) -> scalar``
        (compile-once mode).  Otherwise 1-arg: ``(params) -> scalar``.
    num_inner_steps
        Number of HRSS steps per new particle generation.
    num_delete
        Particles to delete and replace per NS step.
    stepper_fn
        ``(x, direction, t) -> (x_new, is_accepted)``.
    generate_slice_direction_fn
        ``(rng_key, position, **kwargs) -> direction_pytree``.
    init_state_strategy_fn
        Function to initialize NSState from positions.
    update_inner_kernel_params_fn
        ``(rng_key, state, info, params) -> params``.
    delete_fn
        ``(state, num_delete) -> (dead_idx, target_update_idx)``.
    update_strategy
        Strategy for generating replacement particles.
    max_steps
        Maximum expansion steps in slice sampling.
    max_shrinkage
        Maximum shrinking steps in slice sampling.
    data_args : dict, optional
        If provided, enables compile-once mode: ``loglikelihood_fn`` is
        2-arg and ``data_args`` is stored in the state so changing galaxy
        data does not trigger XLA recompilation.

    Returns
    -------
    SamplingAlgorithm
        A (init, step) pair for NSS.
    """
    init_state_fn = partial(
        init_state_strategy_fn,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )

    kernel = build_kernel(
        init_state_fn,
        num_inner_steps,
        num_delete,
        stepper_fn=stepper_fn,
        generate_slice_direction_fn=generate_slice_direction_fn,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        delete_fn=delete_fn,
        update_strategy=update_strategy,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def init_fn(position, data_args=None, rng_key=None):
        """Initialize NSS state from initial positions (optional data for compile-once mode)."""
        if data_args is not None:
            _init_fn = jax.vmap(partial(init_state_fn, data_args=data_args))
        else:
            _init_fn = jax.vmap(init_state_fn)
        state = init(
            position,
            init_state_fn=_init_fn,
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        )
        if data_args is not None:
            state = state._replace(
                inner_kernel_params={**state.inner_kernel_params, "data_args": data_args}
            )
        return state

    def step_fn(rng_key, state):
        """Execute one NSS iteration: delete, resample via HRSS, update covariance."""
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
