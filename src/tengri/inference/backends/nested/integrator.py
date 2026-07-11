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
"""Evidence integration for Nested Sampling.

Tracks the evidence integral during a NS run, computing the marginal
likelihood (evidence), information gain, and related quantities.

Implements the same algorithms as blackjax.ns.integrator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from tengri.inference.backends.nested._types import Array
from tengri.inference.backends.nested.base import StateWithLogLikelihood

__all__ = ["NSIntegrator", "init_integrator", "update_integrator"]


class NSIntegrator(NamedTuple):
    """Integrator for computing the evidence integral.

    Attributes
    ----------
    logX
        Log of the current prior volume estimate.
    logZ
        Accumulated log evidence from dead points.
    logZ_live
        Current estimate of log evidence from live points.
    """

    logX: Array
    logZ: Array
    logZ_live: Array


def init_integrator(particle_state: StateWithLogLikelihood) -> NSIntegrator:
    """Initialize the evidence integrator from initial live points.

    Parameters
    ----------
    particle_state
        Initial live particles.

    Returns
    -------
    NSIntegrator
    """
    ll_dtype = particle_state.loglikelihood.dtype
    logX = jnp.array(0.0, dtype=ll_dtype)
    logZ = jnp.array(-jnp.inf, dtype=ll_dtype)
    logZ_live = _logmeanexp(particle_state.loglikelihood) + logX
    return NSIntegrator(logX, logZ, logZ_live)


def update_integrator(
    integrator: NSIntegrator,
    particle_state: StateWithLogLikelihood,
    dead_particles: StateWithLogLikelihood,
) -> NSIntegrator:
    """Update the evidence integrator after a NS step.

    Parameters
    ----------
    integrator
        Current integrator state.
    particle_state
        Updated live particles after the NS step.
    dead_particles
        Particles that died in this step.

    Returns
    -------
    NSIntegrator
    """
    loglikelihood = particle_state.loglikelihood
    dead_loglikelihood = dead_particles.loglikelihood

    num_particles = len(loglikelihood)
    num_deleted = len(dead_loglikelihood)
    ll_dtype = loglikelihood.dtype
    num_live = jnp.arange(num_particles, num_particles - num_deleted, -1)
    delta_logX = -1.0 / num_live.astype(ll_dtype)
    logX = integrator.logX + jnp.cumsum(delta_logX)
    log_delta_X = logX + jnp.log(1 - jnp.exp(delta_logX))
    log_delta_Z = dead_loglikelihood + log_delta_X

    delta_logZ = logsumexp(log_delta_Z)
    logZ = jnp.logaddexp(integrator.logZ, delta_logZ)
    logZ_live = _logmeanexp(loglikelihood) + logX[-1]
    return NSIntegrator(logX[-1], logZ, logZ_live)


def _logmeanexp(x: Array) -> Array:
    """Compute log(mean(exp(x))) in a numerically stable way."""
    n = jnp.array(x.shape[0])
    return logsumexp(x) - jnp.log(n)
