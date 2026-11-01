# SPDX-License-Identifier: BSD-3-Clause
"""Damped saddle-free Newton steps on a log-density, shared by the inference backends.

The helpers take a log-density (maximized), the convention the samplers use.
The step math runs on the loss ``-logdensity``, so a step is accepted only when
the log-density strictly rises. Used by the Laplace backend to polish the MAP
before the expansion, and by the hierarchical mode search.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def loss_derivatives(logdensity_fn, position, data_args):
    """Gradient and symmetric Hessian of the loss ``-logdensity`` at a point.

    Parameters
    ----------
    logdensity_fn : callable
        ``logdensity_fn(position, data_args) -> scalar``, JAX-traceable.
    position : array_like, shape (n_dim,)
        Flat point at which to evaluate.
    data_args : pytree
        Passed through to ``logdensity_fn``; not differentiated.

    Returns
    -------
    loss_grad : jnp.ndarray, shape (n_dim,)
        Gradient of ``-logdensity_fn`` at ``position``.
    loss_hessian : jnp.ndarray, shape (n_dim, n_dim)
        Symmetrized Hessian of ``-logdensity_fn`` at ``position``.

    Notes
    -----
    JIT/vmap-safe.
    """
    grad = jax.grad(logdensity_fn)(position, data_args)
    hessian = jax.hessian(logdensity_fn)(position, data_args)
    hessian = 0.5 * (hessian + hessian.T)
    return -grad, -hessian


def newton_trial_steps(grad, hessian, *, n_trials=8, damping_growth=10.0):
    """Candidate damped saddle-free Newton steps, least damped first.

    Parameters
    ----------
    grad : array_like, shape (n_dim,)
        Gradient of the loss at the current point.
    hessian : array_like, shape (n_dim, n_dim)
        Hessian of the loss at the current point. Symmetrized before use.
    n_trials : int, optional
        Number of candidate steps. Default 8.
    damping_growth : float, optional
        Ratio between consecutive damping values. Default 10.

    Returns
    -------
    steps : jnp.ndarray, shape (n_trials, n_dim)
        Row ``k`` is the step at damping ``dampings[k]``.
    dampings : jnp.ndarray, shape (n_trials,)
        Damping added to the absolute eigenvalues. The first entry is zero.

    Notes
    -----
    With :math:`H = V \\operatorname{diag}(\\lambda) V^{T}` and
    :math:`s = \\max_i |\\lambda_i|`, the step at damping :math:`\\mu_k` is

    .. math::

        \\Delta x_k = -V\\, \\operatorname{diag}\\!\\left(
        \\frac{1}{|\\lambda| + \\mu_k}\\right) V^{T} g,
        \\qquad \\mu_0 = 0,\\quad \\mu_k = 10^{-3}\\, s\\, c^{\\,k-1},

    where :math:`g` is ``grad`` and :math:`c` is ``damping_growth``. Taking
    absolute eigenvalues is the saddle-free Newton method of Dauphin et al.
    (2014): along a negative-curvature direction the step still descends
    instead of climbing toward the saddle. A zero eigenvalue at
    :math:`\\mu_0 = 0` gives a non-finite step, which
    :func:`first_accepted_step` rejects. JIT/vmap-safe.
    """
    sym = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(sym)
    abs_eigenvalues = jnp.abs(eigenvalues)
    scale = jnp.max(abs_eigenvalues)
    k = jnp.arange(n_trials, dtype=float)
    dampings = jnp.where(k == 0, 0.0, 1e-3 * scale * damping_growth ** (k - 1.0))
    projected = eigenvectors.T @ grad
    denominators = abs_eigenvalues[None, :] + dampings[:, None]
    steps = -(projected[None, :] / denominators) @ eigenvectors.T
    return steps, dampings


def _acceptable(steps, trial_losses, loss0):
    """Per-trial accept predicate shared by every Newton accept path.

    A trial is acceptable when its loss is finite and strictly below ``loss0``
    and its step is entirely finite. Rows of ``steps`` are trials.
    """
    return (
        jnp.isfinite(trial_losses) & (trial_losses < loss0) & jnp.all(jnp.isfinite(steps), axis=1)
    )


def first_accepted_step(x, steps, trial_losses, loss0):
    """Return the first acceptable trial step applied to ``x``.

    Parameters
    ----------
    x : array_like, shape (n_dim,)
        Current point.
    steps : array_like, shape (n_trials, n_dim)
        Candidate steps, least damped first.
    trial_losses : array_like, shape (n_trials,)
        Loss at ``x + steps[k]`` for each trial.
    loss0 : float
        Loss at ``x``.

    Returns
    -------
    x_new : jnp.ndarray, shape (n_dim,)
        ``x + steps[index]`` when a trial is accepted, otherwise ``x`` unchanged.
    accepted : jnp.ndarray, shape ()
        Boolean, ``True`` when a trial was accepted.
    index : jnp.ndarray, shape ()
        Index of the accepted trial (``0`` when none was accepted).

    Notes
    -----
    A trial is accepted when its loss is finite and strictly below ``loss0``
    and its step is finite. Non-finite rows are zeroed before the addition, so
    they cannot leak NaN into ``x_new``. JIT/vmap-safe.
    """
    steps_safe = jnp.where(jnp.isfinite(steps), steps, 0.0)
    ok = _acceptable(steps, trial_losses, loss0)
    index = jnp.argmax(ok)
    accepted = jnp.any(ok)
    x_new = jnp.where(accepted, x + steps_safe[index], x)
    return x_new, accepted, index


def _loss_fn(logdensity_fn, data_args):
    """Flat loss ``x -> -logdensity_fn(x, data_args)``."""

    def loss(x):
        return -logdensity_fn(x, data_args)

    return loss


def _trial_losses(logdensity_fn, positions, data_args):
    """Loss at each row of ``positions``: shape (n_trials, n_dim) to (n_trials,).

    JIT/vmap-safe. Vectorized with ``jax.vmap``.
    """
    return jax.vmap(_loss_fn(logdensity_fn, data_args))(positions)


# Compiled kernels for the eager callers. ``logdensity_fn`` is static, so XLA
# caches one program per log-density object. The samplers cache that object on
# the model, so repeated fits of the same model reuse the compiled code.
_loss_derivatives_jit = jax.jit(loss_derivatives, static_argnums=0)
_trial_losses_jit = jax.jit(_trial_losses, static_argnums=0)


def _step_from(
    position, loss_grad, loss_hessian, loss0, trial_losses_at, *, n_trials, damping_growth
):
    """One damped step from a point whose loss derivatives and loss are already known."""
    steps, _ = newton_trial_steps(
        loss_grad, loss_hessian, n_trials=n_trials, damping_growth=damping_growth
    )
    trial_losses = trial_losses_at(position + steps)
    x_new, accepted, _ = first_accepted_step(position, steps, trial_losses, loss0)
    return x_new, accepted


def damped_newton_step(position, logdensity_fn, data_args, *, n_trials=8, damping_growth=10.0):
    """One damped Newton step on a log-density: ``(new_position, accepted)``.

    Parameters
    ----------
    position : array_like, shape (n_dim,)
        Current flat point.
    logdensity_fn : callable
        ``logdensity_fn(position, data_args) -> scalar`` (maximized), JAX-traceable.
    data_args : pytree
        Passed through to ``logdensity_fn``.
    n_trials : int, optional
        Number of candidate damped steps. Default 8.
    damping_growth : float, optional
        Ratio between consecutive damping values. Default 10.

    Returns
    -------
    new_position : jnp.ndarray, shape (n_dim,)
        The first accepted trial point, or ``position`` unchanged.
    accepted : jnp.ndarray, shape ()
        Boolean, ``True`` when a trial strictly raised the log-density.

    Notes
    -----
    Gradient and Hessian come from :func:`loss_derivatives`; every trial is
    evaluated with ``jax.vmap``; the accept rule is :func:`first_accepted_step`.
    JIT/vmap-safe.
    """
    loss_grad, loss_hessian = loss_derivatives(logdensity_fn, position, data_args)
    loss0 = _loss_fn(logdensity_fn, data_args)(position)
    return _step_from(
        position,
        loss_grad,
        loss_hessian,
        loss0,
        lambda positions: _trial_losses(logdensity_fn, positions, data_args),
        n_trials=n_trials,
        damping_growth=damping_growth,
    )


def newton_polish(
    position, logdensity_fn, data_args, *, max_steps, tol, loss_grad=None, loss_hessian=None
):
    """Take up to ``max_steps`` accepted damped Newton steps toward a maximum.

    Parameters
    ----------
    position : array_like, shape (n_dim,)
        Starting flat point, typically an optimizer's endpoint.
    logdensity_fn : callable
        ``logdensity_fn(position, data_args) -> scalar`` (maximized).
    data_args : pytree
        Passed through to ``logdensity_fn``.
    max_steps : int
        Maximum number of Newton steps.
    tol : float
        Stop when ``max |grad|`` of the loss is at most ``tol``.
    loss_grad, loss_hessian : array_like, optional
        Loss gradient ``(n_dim,)`` and Hessian ``(n_dim, n_dim)`` at ``position``,
        if the caller already has them. When both are given they are used for
        the starting point and the derivatives are not recomputed. Default
        ``None`` computes them.

    Returns
    -------
    position : jnp.ndarray, shape (n_dim,)
        Final point; unchanged when no step was accepted.
    n_accepted : int
        Number of accepted steps.
    loss_grad : jnp.ndarray, shape (n_dim,)
        Gradient of the loss at the final point.
    loss_hessian : jnp.ndarray, shape (n_dim, n_dim)
        Symmetric Hessian of the loss at the final point, so callers reuse it
        instead of recomputing.

    Notes
    -----
    Stops at the first trial round that accepts nothing. Python loop; each
    round is one call to the compiled trial-loss kernel. Derivatives come from
    the compiled :func:`loss_derivatives` kernel.
    """
    position = jnp.asarray(position)
    if loss_grad is None or loss_hessian is None:
        loss_grad, loss_hessian = _loss_derivatives_jit(logdensity_fn, position, data_args)
    n_accepted = 0
    for _ in range(max_steps):
        if float(jnp.max(jnp.abs(loss_grad))) <= tol:
            break
        loss0 = float(_trial_losses_jit(logdensity_fn, position[None, :], data_args)[0])
        new_position, accepted = _step_from(
            position,
            loss_grad,
            loss_hessian,
            loss0,
            lambda positions: _trial_losses_jit(logdensity_fn, positions, data_args),
            n_trials=8,
            damping_growth=10.0,
        )
        if not bool(accepted):
            break
        position = new_position
        n_accepted += 1
        loss_grad, loss_hessian = _loss_derivatives_jit(logdensity_fn, position, data_args)
    return position, n_accepted, loss_grad, loss_hessian
