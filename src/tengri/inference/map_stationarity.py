# SPDX-License-Identifier: BSD-3-Clause
"""Stationarity of a MAP point: gradient, Hessian and Newton decrement.

One measure for both MAP paths and for the tests that certify them. The point is
stationary when its Hessian is positive definite and its Newton decrement is below
:data:`MAP_DECREMENT_BAR`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

#: Newton-decrement bar [nats] below which a MAP point counts as stationary (#3205).
MAP_DECREMENT_BAR = 1e-8


@dataclass(frozen=True)
class Stationarity:
    """Value, gradient and curvature of a negative log-posterior at one point.

    Attributes
    ----------
    nlp : float
        Negative log-posterior at the point [dimensionless, nats].
    grad_norm : float
        Euclidean norm of the gradient in standardized coordinates [dimensionless].
    decrement : float
        Newton decrement :math:`\\tfrac12 g^\\top H^{-1} g` over the positive-curvature
        eigendirections [nats].
    positive_definite : bool
        True when every Hessian eigenvalue is positive.
    """

    nlp: float
    grad_norm: float
    decrement: float
    positive_definite: bool

    @property
    def stationary(self) -> bool:
        """True when the Hessian is positive definite and the decrement is below the bar."""
        return self.positive_definite and self.decrement < MAP_DECREMENT_BAR


def stationarity(loss: Callable[[jax.Array], jax.Array], x: jax.Array) -> Stationarity:
    """Measure the stationarity of ``loss`` at the flat standardized point ``x``.

    Parameters
    ----------
    loss : callable
        ``loss(vec) -> scalar``, the negative log-posterior of a flat standardized
        vector, ``vec`` of shape ``(D,)``.
    x : array_like, shape (D,)
        Standardized point at which to measure. Values are O(1) by construction, so
        float32 runs represent the gradient and Hessian without scale issues.

    Returns
    -------
    Stationarity
        Value, gradient norm, Newton decrement and curvature sign at ``x``.

    Notes
    -----
    Not JIT-compatible as a whole (it reads concrete eigenvalue signs to report
    ``positive_definite``); the value, gradient and Hessian are each jitted. The
    decrement is summed over the positive-curvature eigendirections only, so a
    non-positive-definite point reports a finite decrement next to
    ``positive_definite=False``.
    """
    x = jnp.asarray(x)
    value, grad = jax.jit(jax.value_and_grad(loss))(x)
    hessian = jax.jit(jax.hessian(loss))(x)
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    positive = eigenvalues > 0
    safe = jnp.where(positive, eigenvalues, 1.0)
    projected = (eigenvectors.T @ grad) ** 2
    decrement = 0.5 * jnp.sum(jnp.where(positive, projected / safe, 0.0))
    return Stationarity(
        nlp=float(value),
        grad_norm=float(jnp.linalg.norm(grad)),
        decrement=float(decrement),
        positive_definite=bool(jnp.all(positive)),
    )
