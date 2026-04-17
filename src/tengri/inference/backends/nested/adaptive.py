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
"""Adaptive Nested Sampling.

Combines SMC-equivalent adaptive tempering and inner kernel tuning.
Ported from blackjax.ns.adaptive.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.inference.backends.nested._types import ArrayLikeTree, ArrayTree, PRNGKey
from tengri.inference.backends.nested.base import (
    NSInfo,
    NSState,
    StateWithLogLikelihood,
    build_kernel as base_build_kernel,
    init as base_init,
)
from tengri.inference.backends.nested.integrator import (
    NSIntegrator,
    init_integrator,
    update_integrator,
)

__all__ = ["build_kernel", "init"]


class AdaptiveNSState(NamedTuple):
    """Extension of NSState with inner kernel parameters and integrator.

    Attributes
    ----------
    particles
        The StateWithLogLikelihood of the current live particles.
    integrator
        The NSIntegrator tracking evidence statistics.
    inner_kernel_params
        Parameters for the inner kernel (e.g., covariance matrix).
    """

    particles: StateWithLogLikelihood
    integrator: NSIntegrator
    inner_kernel_params: dict[str, ArrayTree]


def init(
    positions: ArrayLikeTree,
    init_state_fn: Callable,
    loglikelihood_birth: float = jnp.nan,
    update_inner_kernel_params_fn: Callable | None = None,
    rng_key: jax.random.PRNGKey | None = None,
) -> AdaptiveNSState:
    """Initialize the adaptive NS state.

    Parameters
    ----------
    positions
        Initial positions drawn from the prior.
    init_state_fn
        Function that initializes state from positions.
    loglikelihood_birth
        Initial log-likelihood birth threshold.
    update_inner_kernel_params_fn
        Function to compute initial inner kernel parameters.
    rng_key
        PRNG key for inner kernel parameter initialization.

    Returns
    -------
    AdaptiveNSState
    """
    base_state = base_init(positions, init_state_fn, loglikelihood_birth=loglikelihood_birth)
    integrator = init_integrator(base_state.particles)
    inner_kernel_params = {}
    if update_inner_kernel_params_fn is not None:
        inner_kernel_params = update_inner_kernel_params_fn(rng_key, base_state, None, {})
    return AdaptiveNSState(
        particles=base_state.particles,
        inner_kernel_params=inner_kernel_params,
        integrator=integrator,
    )


def build_kernel(
    delete_fn: Callable,
    inner_kernel: Callable,
    update_inner_kernel_params_fn: Callable[
        [PRNGKey, NSState, NSInfo, dict[str, ArrayTree]], dict[str, ArrayTree]
    ],
) -> Callable:
    """Build an adaptive Nested Sampling kernel.

    Wraps the base NS kernel with adaptive tuning of inner kernel parameters
    and evidence integration.

    Parameters
    ----------
    delete_fn
        ``(state) -> (dead_idx, target_update_idx)``.
    inner_kernel
        ``(rng_key, state, loglikelihood_0, **params) -> (new_particles, info)``.
    update_inner_kernel_params_fn
        ``(rng_key, state, info, params) -> params``.

    Returns
    -------
    Callable
        ``(rng_key, state) -> (AdaptiveNSState, NSInfo)``.
    """

    def kernel(rng_key: PRNGKey, state: AdaptiveNSState) -> tuple[AdaptiveNSState, NSInfo]:
        adapted_kernel = base_build_kernel(
            delete_fn,
            partial(inner_kernel, **state.inner_kernel_params),
        )

        new_state, info = adapted_kernel(rng_key, state)
        inner_kernel_update_key, rng_key = jax.random.split(rng_key)
        new_inner_kernel_params = update_inner_kernel_params_fn(
            inner_kernel_update_key, new_state, info, new_state.inner_kernel_params
        )
        new_integrator_state = update_integrator(
            state.integrator, new_state.particles, info.particles
        )
        return (
            AdaptiveNSState(
                particles=new_state.particles,
                inner_kernel_params=new_inner_kernel_params,
                integrator=new_integrator_state,
            ),
            info,
        )

    return kernel
