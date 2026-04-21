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
"""Particle statistics for tuning inner kernels.

Ported from blackjax.smc.tuning.from_particles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.inference.backends.nested._types import Array

__all__ = [
    "inverse_mass_matrix_from_particles",
    "particles_covariance_matrix",
    "particles_means",
    "particles_stds",
]


def particles_as_rows(particles):
    """Represent multi-variable particles as a matrix (n_particles, n_dims)."""
    return jax.vmap(lambda x: ravel_pytree(x)[0])(particles)


def particles_stds(particles):
    """Compute standard deviation along each dimension of particles.

    Parameters
    ----------
    particles
        PyTree of particles (leading axis is the particle ensemble).

    Returns
    -------
    ndarray, shape (n_dims,)
        Standard deviation per dimension.
    """
    return jnp.std(particles_as_rows(particles), axis=0)


def particles_means(particles):
    """Compute mean along each dimension of particles.

    Parameters
    ----------
    particles
        PyTree of particles (leading axis is the particle ensemble).

    Returns
    -------
    ndarray, shape (n_dims,)
        Mean per dimension.
    """
    return jnp.mean(particles_as_rows(particles), axis=0)


def particles_covariance_matrix(particles):
    """Compute empirical covariance matrix from particle ensemble.

    Parameters
    ----------
    particles
        PyTree of particles (leading axis is the particle ensemble).

    Returns
    -------
    ndarray, shape (n_dims, n_dims)
        Covariance matrix.
    """
    return jnp.cov(particles_as_rows(particles), ddof=0, rowvar=False)


def inverse_mass_matrix_from_particles(particles) -> Array:
    """Inverse mass matrix from particle variances (section 3.1, arXiv:1808.07730)."""
    return jnp.diag(jnp.var(particles_as_rows(particles), axis=0))
