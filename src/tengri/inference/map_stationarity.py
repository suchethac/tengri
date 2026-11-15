# SPDX-License-Identifier: BSD-3-Clause
"""Stationarity of a MAP point: gradient, Hessian and Newton decrement.

One measure for both MAP paths and for the tests that certify them. The point is
stationary when its Hessian is positive definite and its Newton decrement is below
:data:`MAP_DECREMENT_BAR`.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

#: Newton-decrement bar [nats] below which a MAP point counts as stationary (#3205).
MAP_DECREMENT_BAR = 1e-8

#: Jitted (value-and-gradient, Hessian) pairs keyed weakly on the loss, so a repeated
#: certificate on one objective reuses its compiled programs. The jitted pair holds the
#: loss only through a weak reference, so a dropped loss frees its entry.
_DERIVATIVES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def compiled_derivatives(loss: Callable):
    """Return the jitted ``(value_and_grad, hessian)`` pair of ``loss``, built once.

    Parameters
    ----------
    loss : callable
        ``loss(vec) -> scalar``, or ``loss(vec, data) -> scalar`` when the data is passed
        to the returned callables.

    Returns
    -------
    value_and_grad : callable
        ``value_and_grad(x, data=None) -> (value, grad)``, jitted.
    hessian : callable
        ``hessian(x, data=None) -> ndarray, shape (D, D)``, jitted.

    Notes
    -----
    Keyed on the identity of ``loss``: pass the same object to reuse the compiled
    programs, a new object compiles afresh. The entry is dropped with ``loss``.

    ``data`` is a traced argument, not a closure constant, so one compile serves every
    dataset of the same shape (#3530). ``data=None`` calls ``loss(vec)``.
    """
    pair = _DERIVATIVES.get(loss)
    if pair is None:
        loss_ref = weakref.ref(loss)

        def live(vec, data):
            f = loss_ref()
            if f is None:
                raise RuntimeError("the loss was collected while its derivatives were in use")
            return f(vec) if data is None else f(vec, data)

        pair = (jax.jit(jax.value_and_grad(live)), jax.jit(jax.hessian(live)))
        _DERIVATIVES[loss] = pair
    value_and_grad, hessian = pair
    return (
        lambda x, data=None: value_and_grad(x, data),
        lambda x, data=None: hessian(x, data),
    )


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
    hessian_finite : bool
        True when every Hessian entry is finite. False is a distinct status: the
        curvature is unknown, so ``decrement`` is NaN and ``positive_definite`` is
        False, and the point cannot be certified.
    """

    nlp: float
    grad_norm: float
    decrement: float
    positive_definite: bool
    hessian_finite: bool

    @property
    def stationary(self) -> bool:
        """True when the Hessian is finite, positive definite and decrement is below the bar."""
        return (
            self.hessian_finite and self.positive_definite and self.decrement < MAP_DECREMENT_BAR
        )


def stationarity(loss: Callable, x: jax.Array, data=None) -> Stationarity:
    """Measure the stationarity of ``loss`` at the flat standardized point ``x``.

    Parameters
    ----------
    loss : callable
        ``loss(vec) -> scalar``, the negative log-posterior of a flat standardized
        vector, ``vec`` of shape ``(D,)``; or ``loss(vec, data)`` when ``data`` is given.
    x : array_like, shape (D,)
        Standardized point at which to measure. Values are O(1) by construction, so
        float32 runs represent the gradient and Hessian without scale issues.
    data : pytree, optional
        Passed to ``loss`` as a traced argument (see :func:`compiled_derivatives`).

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
    ``positive_definite=False``. A non-finite Hessian is reported as
    ``hessian_finite=False`` with a NaN decrement, never as a zero decrement.
    """
    x = jnp.asarray(x)
    value_and_grad, hessian_fn = compiled_derivatives(loss)
    value, grad = value_and_grad(x, data)
    hessian = hessian_fn(x, data)
    grad_norm = float(jnp.linalg.norm(grad))
    if not bool(jnp.all(jnp.isfinite(hessian))):
        return Stationarity(
            nlp=float(value),
            grad_norm=grad_norm,
            decrement=float("nan"),
            positive_definite=False,
            hessian_finite=False,
        )
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    positive = eigenvalues > 0
    safe = jnp.where(positive, eigenvalues, 1.0)
    projected = (eigenvectors.T @ grad) ** 2
    decrement = 0.5 * jnp.sum(jnp.where(positive, projected / safe, 0.0))
    return Stationarity(
        nlp=float(value),
        grad_norm=grad_norm,
        decrement=float(decrement),
        positive_definite=bool(jnp.all(positive)),
        hessian_finite=True,
    )
