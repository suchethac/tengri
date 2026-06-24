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
"""Hit-and-Run Slice Sampling.

Implements the Hit-and-Run Slice Sampling algorithm (Neal 2003).
Ported from blackjax.mcmc.ss.

References
----------
.. [1] Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.inference.backends.nested._types import (
    Array,
    ArrayLikeTree,
    ArrayTree,
    PRNGKey,
    SamplingAlgorithm,
)

__all__ = [
    "SliceInfo",
    "SliceState",
    "build_hrss_kernel",
    "build_kernel",
    "hrss_as_top_level_api",
    "init",
    "sample_direction_from_covariance",
]


class SliceState(NamedTuple):
    """State of the Slice Sampling algorithm.

    Attributes
    ----------
    position
        The current position of the chain.
    logdensity
        The log-density of the target distribution at the current position.
    """

    position: ArrayLikeTree
    logdensity: float


class SliceInfo(NamedTuple):
    """Additional information about the Slice Sampling transition.

    Attributes
    ----------
    is_accepted
        A boolean indicating whether the proposed sample was accepted.
    num_steps
        The number of steps taken to expand the interval during the stepping-out phase.
    num_shrink
        The number of steps taken during the shrinking phase to find an
        acceptable sample.
    """

    is_accepted: bool
    num_steps: int
    num_shrink: int


def init(position: ArrayTree, logdensity_fn: Callable) -> SliceState:
    """Initialize the Slice Sampler state.

    Parameters
    ----------
    position
        The initial position of the chain.
    logdensity_fn
        A function that computes the log-density of the target distribution.

    Returns
    -------
    SliceState
        The initial state of the Slice Sampler.
    """
    return SliceState(position, logdensity_fn(position))


def build_kernel(
    slice_fn: Callable[[float], tuple[SliceState, bool]],
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build a Slice Sampling kernel.

    Parameters
    ----------
    slice_fn
        A function that takes a scalar parameter ``t`` and returns a tuple
        (SliceState, is_accepted) indicating the state at that parameter value
        and whether it satisfies acceptance criteria.
    max_steps
        Maximum number of steps when expanding the interval in each direction
        during the stepping-out phase.
    max_shrinkage
        Maximum number of shrinking steps to avoid infinite loops.

    Returns
    -------
    Callable
        A kernel function: (rng_key, SliceState) -> (SliceState, SliceInfo).

    References
    ----------
    .. [1] Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.
    """

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
    ) -> tuple[SliceState, SliceInfo]:
        """Execute slice sampling: vertical sample, then horizontal expansion/shrinking."""
        vs_key, hs_key = jax.random.split(rng_key)
        u = jax.random.uniform(vs_key)
        logslice = state.logdensity + jnp.log(u)
        vertical_is_accepted = logslice < state.logdensity

        def _slice_fn(t):
            """Evaluate slice_fn at t, accepting only if log density exceeds the slice level."""
            new_state, is_accepted = slice_fn(t)
            in_slice = new_state.logdensity >= logslice
            return new_state, is_accepted & in_slice

        new_state, info = horizontal_slice(hs_key, state, _slice_fn, max_steps, max_shrinkage)
        info = info._replace(is_accepted=info.is_accepted & vertical_is_accepted)
        return new_state, info

    return kernel


def horizontal_slice(
    rng_key: PRNGKey,
    state: SliceState,
    slice_fn: Callable[[float], tuple[SliceState, bool]],
    m: int,
    max_shrinkage: int,
) -> tuple[SliceState, SliceInfo]:
    """Propose a new sample using stepping-out and shrinking procedures.

    First expands an interval [l, r] along a 1D parameterization until both
    ends are outside the slice (stepping-out), then samples uniformly and
    shrinks until an acceptable point is found.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    state
        The current slice sampling state.
    slice_fn
        ``(t) -> (SliceState, is_accepted)``.
    m
        Maximum number of expansion steps in each direction.
    max_shrinkage
        Maximum number of shrinking steps.

    Returns
    -------
    tuple[SliceState, SliceInfo]
    """
    # Initial bounds
    rng_key, subkey = jax.random.split(rng_key)
    u, v = jax.random.uniform(subkey, (2,))
    j = jnp.floor(m * v).astype(jnp.int32)
    k = (m - 1) - j

    # Expand
    def step_body_fun(carry):
        """Expand interval in one direction: step and check acceptance."""
        i, s, t, _ = carry
        t += s
        _, is_accepted = slice_fn(t)
        i -= 1
        return i, s, t, is_accepted

    def step_cond_fun(carry):
        """Continue stepping if within bounds and point is outside slice."""
        i, _, _, is_accepted = carry
        return is_accepted & (i > 0)

    j, _, left, _ = jax.lax.while_loop(step_cond_fun, step_body_fun, (j + 1, -1, 1 - u, True))
    k, _, right, _ = jax.lax.while_loop(step_cond_fun, step_body_fun, (k + 1, +1, -u, True))

    # Shrink
    def shrink_body_fun(carry):
        """Shrink interval: sample uniformly, accept or update bounds."""
        n, rng_key, left, right, _state, is_accepted = carry

        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, minval=left, maxval=right)

        new_state, is_accepted = slice_fn(u)
        n += 1

        left = jnp.where(u < 0, u, left)
        right = jnp.where(u > 0, u, right)

        return n, rng_key, left, right, new_state, is_accepted

    def shrink_cond_fun(carry):
        """Continue shrinking if proposal rejected and iteration limit not reached."""
        n, _, _, _, _, is_accepted = carry
        return ~is_accepted & (n < max_shrinkage)

    carry = (0, rng_key, left, right, state, False)
    carry = jax.lax.while_loop(shrink_cond_fun, shrink_body_fun, carry)
    n, _, _, _, new_state, is_accepted = carry
    new_state = jax.tree.map(lambda new, old: jnp.where(is_accepted, new, old), new_state, state)
    slice_info = SliceInfo(is_accepted, m + 1 - j - k, n)
    return new_state, slice_info


def sample_direction_from_covariance(
    rng_key: PRNGKey, position: ArrayLikeTree, cov: Array
) -> Array:
    """Generate a random direction vector from a multivariate Gaussian.

    The direction is Mahalanobis-normalized and scaled by 2.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    position
        Current position (used for shape extraction).
    cov
        The covariance matrix.

    Returns
    -------
    ArrayTree
        A normalized direction vector (same pytree structure as position).
    """
    _p, unravel_fn = ravel_pytree(position)
    d = jax.random.multivariate_normal(rng_key, mean=jnp.zeros(cov.shape[0]), cov=cov)
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("...i,...ij,...j", d, invcov, d))
    d = d / norm[..., None]
    d *= 2
    return unravel_fn(d)


def build_hrss_kernel(
    cov: Array,
    init_fn: Callable = init,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build a Hit-and-Run Slice Sampling kernel.

    Parameters
    ----------
    cov
        Covariance matrix for the direction proposal.
    init_fn
        Function initializing a SliceState.
    max_steps
        Maximum expansion steps per direction.
    max_shrinkage
        Maximum shrinking steps.

    Returns
    -------
    Callable
        Kernel: (rng_key, SliceState, logdensity_fn) -> (SliceState, SliceInfo).
    """

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable
    ) -> tuple[SliceState, SliceInfo]:
        """Run one HRSS iteration: sample a direction from the covariance, then slice along it."""
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = sample_direction_from_covariance(prop_key, state.position, cov)

        def slice_fn(t):
            """Evaluate the logdensity at position displaced by t along direction d."""
            x = jax.tree.map(lambda x, d: x + t * d, state.position, d)
            is_accepted = True
            new_state = init_fn(x, logdensity_fn)
            return new_state, is_accepted

        slice_kernel = build_kernel(slice_fn, max_steps, max_shrinkage)
        return slice_kernel(rng_key, state)

    return kernel


def hrss_as_top_level_api(
    logdensity_fn: Callable,
    cov: Array,
    init_fn: Callable = init,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Create a Hit-and-Run Slice Sampling algorithm.

    Parameters
    ----------
    logdensity_fn
        Log-density function of the target distribution.
    cov
        Covariance matrix for direction proposals.
    init_fn
        Function initializing a SliceState.
    max_steps
        Maximum expansion steps per direction.
    max_shrinkage
        Maximum shrinking steps.

    Returns
    -------
    SamplingAlgorithm
        A (init, step) pair for HRSS.
    """
    kernel = build_hrss_kernel(cov, init_fn, max_steps, max_shrinkage)
    init_fn = partial(init_fn, logdensity_fn=logdensity_fn)
    step_fn = partial(kernel, logdensity_fn=logdensity_fn)
    return SamplingAlgorithm(init_fn, step_fn)
