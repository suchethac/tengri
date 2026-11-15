# SPDX-License-Identifier: BSD-3-Clause
"""Optax L-BFGS run to a gradient tolerance: the driver both MAP paths share.

The batched spectroscopy MAP vmaps this over galaxies, and the per-galaxy
``Fitter`` MAP continues from scipy's L-BFGS-B stop with it when scipy stops short
(#3205). One driver keeps the two paths on the same stopping rule.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class BoundLoss:
    """``loss(params, data)`` with the data as a pytree child, so it is traced, not closed over.

    The loss is the static part of the tree, so one compiled driver serves every dataset
    of the same shape: the data is a traced argument of the jitted driver (#3530).
    ``data=None`` calls ``loss(params)``.
    """

    def __init__(self, loss: Callable, data: Any = None):
        self.loss = loss
        self.data = data

    def __call__(self, params):
        return self.loss(params) if self.data is None else self.loss(params, self.data)

    def tree_flatten(self):
        return (self.data,), self.loss

    @classmethod
    def tree_unflatten(cls, loss, children):
        return cls(loss, children[0])


def max_abs_leaf(tree: Any) -> jax.Array:
    """Largest absolute entry over every leaf of a pytree.

    Parameters
    ----------
    tree : pytree of array_like
        Gradient pytree in the optimizer's coordinates.

    Returns
    -------
    jax.Array, shape ()
        ``max |leaf|`` over all leaves.
    """
    return jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]))


#: Jitted drivers keyed on ``n_steps``. Within one, JAX keys the compiled program on the
#: loss (the static part of a :class:`BoundLoss`), the data shapes and the initial point.
_COMPILED_DRIVERS: dict[int, Callable] = {}


def _compiled_driver(n_steps: int) -> Callable:
    driver = _COMPILED_DRIVERS.get(n_steps)
    if driver is None:
        driver = jax.jit(lambda bound, init, gtol: _lbfgs_to_gtol(bound, init, n_steps, gtol))
        _COMPILED_DRIVERS[n_steps] = driver
    return driver


def lbfgs_to_gtol(
    value_fn: Callable[[Any], jax.Array],
    init: Any,
    *,
    n_steps: int,
    gtol: float,
) -> tuple[Any, jax.Array, jax.Array]:
    """Run optax L-BFGS from ``init`` until ``max |grad| < gtol`` or ``n_steps``.

    Parameters
    ----------
    value_fn : callable
        ``value_fn(params) -> scalar``, the objective to minimize. Traceable. A
        :class:`BoundLoss` is run through a compiled driver, so its data is traced.
    init : pytree of array_like
        Starting parameters in the optimizer's (standardized) coordinates.
    n_steps : int
        Iteration cap.
    gtol : float
        Stop when the largest absolute gradient entry is below this value.

    Returns
    -------
    params : pytree
        Final parameters.
    value : jax.Array, shape ()
        Objective at ``params``.
    trace : jax.Array, shape (n_steps,)
        Objective at the start of each iteration. Entries after the stop hold the
        final value.

    Notes
    -----
    **JIT-compatible** and **vmap-safe**: the loop is a ``lax.while_loop``, so a
    vmapped call stops each batch member at its own gradient tolerance. Uses the
    zoom line search of ``optax.lbfgs``.

    A :class:`BoundLoss` runs through a jitted driver, so repeated calls on one loss
    (the continuation rounds of a certificate, and later runs) compile once (#3530).
    """
    if isinstance(value_fn, BoundLoss):
        return _compiled_driver(n_steps)(value_fn, init, gtol)
    return _lbfgs_to_gtol(value_fn, init, n_steps, gtol)


def _lbfgs_to_gtol(
    value_fn: Callable[[Any], jax.Array], init: Any, n_steps: int, gtol: Any
) -> tuple[Any, jax.Array, jax.Array]:
    """The optax L-BFGS loop of :func:`lbfgs_to_gtol`, traceable as is."""
    import optax

    optimizer = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(value_fn)
    state = optimizer.init(init)
    value, grad = value_and_grad(init, state=state)

    def not_converged(carry):
        _, _, _, grad_now, it, _ = carry
        return (it < n_steps) & jnp.logical_not(max_abs_leaf(grad_now) < gtol)

    def advance(carry):
        params, opt_state, value_now, grad_now, it, trace = carry
        trace = trace.at[it].set(value_now)
        updates, opt_state = optimizer.update(
            grad_now, opt_state, params, value=value_now, grad=grad_now, value_fn=value_fn
        )
        params = optax.apply_updates(params, updates)
        value_next, grad_next = value_and_grad(params, state=opt_state)
        return params, opt_state, value_next, grad_next, it + 1, trace

    carry = (init, state, value, grad, jnp.asarray(0), jnp.zeros((n_steps,), dtype=value.dtype))
    params, _, value, _, n_done, trace = jax.lax.while_loop(not_converged, advance, carry)
    trace = jnp.where(jnp.arange(n_steps) < n_done, trace, value)
    return params, value, trace
