# SPDX-License-Identifier: BSD-3-Clause
"""Laplace approximation: Gaussian posterior from Hessian at MAP.

The cheapest uncertainty estimate available — compute the Hessian of the
loss function H(xi) at the MAP estimate, invert to get a covariance matrix,
then draw samples from N(theta_MAP, H^{-1}).

Also provides a Laplace evidence estimate:
    log Z_laplace = -H(theta_MAP) + (D/2)*log(2*pi) - 0.5*log(det(H))

Works entirely in unbounded parameter space (where the loss is smooth),
then transforms samples to physical space.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tengri.config.exceptions import LaplaceNotAtModeWarning, warn_measured
from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical

# Newton decrement above which the expansion point is reported as off-mode
# [nats]. An offset of ``delta`` standard deviations along a single direction
# gives ``d = delta^2 / 2``, so this is ~0.45 sigma. Measured on a 64-galaxy
# DRW-field population: converged fits scored 0.0005-0.075, under-converged
# ones 1.3 upward (issue #1537).
DEFAULT_STATIONARITY_TOL = 0.1


def _finite_diff_hessian(grad_fn, theta_flat, unravel_fn, data_args, eps=1e-5):
    """Hessian via central finite differences on pre-compiled grad_fn.

    2×D gradient evaluations using the already-JIT-compiled grad_fn.
    Zero additional JAX compilation — each call is a cache hit on the
    same compiled kernel.

    For D=7 at ~350μs/eval: total ≈ 5ms (vs 55s for ``jax.hessian``
    which compiles a monolithic second-derivative XLA program).
    """
    n = len(theta_flat)
    H = np.zeros((n, n))

    for j in range(n):
        h = eps * max(abs(float(theta_flat[j])), 1.0)

        theta_plus = theta_flat.at[j].set(theta_flat[j] + h)
        theta_minus = theta_flat.at[j].set(theta_flat[j] - h)

        _, grad_plus = grad_fn(unravel_fn(theta_plus), data_args)
        _, grad_minus = grad_fn(unravel_fn(theta_minus), data_args)

        gp_flat, _ = ravel_pytree(grad_plus)
        gm_flat, _ = ravel_pytree(grad_minus)

        H[:, j] = np.asarray((gp_flat - gm_flat) / (2.0 * h))

    return jnp.asarray(H)


def _newton_decrement(grad_flat, eigenvalues, eigenvectors):
    """Loss drop a quadratic model predicts between here and the mode [nats].

    ``d = 0.5 g^T H^-1 g``, evaluated in the Hessian eigenbasis so no second
    inverse is needed. Zero exactly at a stationary point, and invariant under
    affine reparameterization — unlike ``||g||``, which rescales with the
    parameters and so cannot carry a fixed threshold.

    Parameters
    ----------
    grad_flat : array_like, shape (n_dim,)
        Gradient of the loss at the expansion point, in unbounded space.
    eigenvalues : array_like, shape (n_dim,)
        Hessian eigenvalues, already floored to be positive.
    eigenvectors : array_like, shape (n_dim, n_dim)
        Corresponding eigenvectors, as columns.

    Returns
    -------
    float
        Predicted loss drop [nats]. An offset of ``delta`` standard deviations
        along one direction gives ``delta**2 / 2``. ``inf`` when any eigenvalue
        is non-positive: the quadratic model then has no minimum to descend to,
        and the point cannot be a mode whatever the gradient does.

    Notes
    -----
    The non-positive case is not a corner case to tidy away — it is the one
    input that breaks the formula's own algebra. A negative eigenvalue makes
    the sum **negative**, so a naive ``d > tol`` test reads as "converged" at a
    saddle. Reachable whenever ``regularize=False`` leaves the spectrum
    unfloored.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if not bool(jnp.all(eigenvalues > 0)):
        return float("inf")
    projected = eigenvectors.T @ grad_flat
    return float(0.5 * jnp.sum(projected**2 / eigenvalues))


def run_laplace(
    *,
    key,
    loss_fn,
    data_args,
    map_params_unbounded,
    to_physical_fn,
    model,
    grad_fn=None,
    n_samples=2000,
    regularize=True,
    min_eigenvalue=1e-6,
    stationarity_tol=DEFAULT_STATIONARITY_TOL,
    verbose=True,
):
    """Run Laplace approximation from a MAP estimate.

    Parameters
    ----------
    key : PRNGKey
        Random key for sampling.
    loss_fn : callable
        Loss function: ``(unbounded param dict, data_args) -> scalar``.
    data_args : dict
        Observed data dict (``data``, ``noise``, etc.).
    map_params_unbounded : dict
        MAP parameters in unbounded space.
    to_physical_fn : callable
        Converts unbounded param dict to physical space.
    model : Model
        Forward model (stored in Posterior).
    grad_fn : callable, optional
        Pre-compiled ``(params, data_args) -> (loss, grad)`` function.
        When provided, the Hessian is computed via central finite
        differences — avoiding the monolithic ``jax.hessian`` compilation
        (55s → 5ms for D=7).  Falls back to ``jax.hessian`` if ``None``.
    n_samples : int
        Number of posterior samples to draw.
    regularize : bool
        Clip small Hessian eigenvalues to ensure positive definiteness.
    min_eigenvalue : float
        Minimum eigenvalue threshold (only if regularize=True).
    stationarity_tol : float
        Newton decrement [nats] above which the expansion point is reported as
        off-mode via :class:`~tengri.config.exceptions.LaplaceNotAtModeWarning`.
        Default 0.1, an offset of ~0.45 standard deviations. Raise it to
        silence the check; the decrement is reported in ``diagnostics`` either
        way.
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        Samples from the Gaussian approximation, with Laplace log-evidence.

    Warns
    -----
    LaplaceNotAtModeWarning
        When the Newton decrement exceeds ``stationarity_tol``, meaning
        ``map_params_unbounded`` is not a stationary point of ``loss_fn`` and
        ``H^-1`` is therefore not a covariance (issue #1537).

    Notes
    -----
    ``cov = H^-1`` holds only at a mode. ``run_map`` takes a fixed number of
    Adam steps with no convergence test, so an under-converged expansion point
    reaches here routinely and produces a confident, plausible, wrong
    posterior — typically far too narrow, since a point on a steep slope
    carries much higher curvature than the mode below it. The Newton decrement

    .. math::

        d = \\tfrac{1}{2}\\, g^{T} H^{-1} g

    measures this, where :math:`g` is the loss gradient at the expansion point
    and :math:`H` the (eigenvalue-floored) Hessian; :math:`d` is in nats and is
    zero exactly at a mode.
    """
    from tengri.inference.posterior import Posterior

    t0 = time.time()

    # Flatten MAP params to a single vector
    theta_flat, unravel_fn = ravel_pytree(map_params_unbounded)
    n_dim = len(theta_flat)

    if verbose:
        print(f"Laplace: {n_dim} parameters, {n_samples} samples")

    # Hessian at MAP — finite differences on pre-compiled grad_fn (fast)
    # or jax.hessian (slow compilation, exact).
    if grad_fn is not None:
        hessian = _finite_diff_hessian(
            grad_fn,
            theta_flat,
            unravel_fn,
            data_args,
        )
    else:

        def loss_flat(x):
            """Evaluate loss on flat unbounded parameter vector."""
            return loss_fn(unravel_fn(x), data_args)

        hessian = jax.hessian(loss_flat)(theta_flat)

    # Symmetrize (numerical safety)
    hessian = 0.5 * (hessian + hessian.T)

    # Eigendecompose for regularization and diagnostics
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    n_clipped = 0

    if regularize:
        n_clipped = int(jnp.sum(eigenvalues < min_eigenvalue))
        eigenvalues_clipped = jnp.maximum(eigenvalues, min_eigenvalue)
        hessian_reg = eigenvectors @ jnp.diag(eigenvalues_clipped) @ eigenvectors.T
    else:
        eigenvalues_clipped = eigenvalues
        hessian_reg = hessian

    # Covariance = H^{-1}
    cov = jnp.linalg.inv(hessian_reg)

    # Symmetrize covariance (numerical safety)
    cov = 0.5 * (cov + cov.T)

    if verbose and n_clipped > 0:
        print(f"  Regularized: {n_clipped}/{n_dim} eigenvalues clipped")

    # Is the expansion point actually a mode?  cov = H^-1 only means covariance
    # there; nothing upstream guarantees it, and nothing downstream can detect
    # it (i.i.d. Gaussian draws score R-hat ~ 1 whatever the shape).
    if grad_fn is not None:
        _, grad_at_map = grad_fn(map_params_unbounded, data_args)
    else:
        grad_at_map = jax.grad(lambda p: loss_fn(p, data_args))(map_params_unbounded)
    grad_flat, _ = ravel_pytree(grad_at_map)

    grad_norm = float(jnp.linalg.norm(grad_flat))
    decrement = _newton_decrement(grad_flat, eigenvalues_clipped, eigenvectors)

    indefinite = not bool(jnp.all(eigenvalues_clipped > 0))
    if indefinite:
        n_bad = int(jnp.sum(eigenvalues_clipped <= 0))
        warn_measured(
            f"Laplace Hessian is indefinite at the expansion point: "
            f"{n_bad}/{n_dim} eigenvalues are non-positive (most negative "
            f"{float(jnp.min(eigenvalues_clipped)):.4g}), gradient norm "
            f"{grad_norm:.4g}. The loss curves *downward* along those "
            f"directions, so this is a saddle or a maximum, not a mode, and "
            f"H^-1 is not a covariance — the draws are meaningless rather "
            f"than merely mis-scaled. Leave regularize=True (the default) to "
            f"floor the spectrum at min_eigenvalue, or re-run the optimizer "
            f"from a different start.",
            LaplaceNotAtModeWarning,
            stacklevel=2,
            grad_norm=grad_norm,
            newton_decrement=decrement,
            n_nonpositive_eigenvalues=n_bad,
            min_eigenvalue=float(jnp.min(eigenvalues_clipped)),
        )
    elif decrement > stationarity_tol:
        warn_measured(
            f"Laplace expansion point is not a mode: Newton decrement "
            f"{decrement:.4g} nats (tolerance {stationarity_tol:g}), gradient "
            f"norm {grad_norm:.4g}. The loss still drops by ~{decrement:.4g} "
            f"nats toward the true mode, so H^-1 describes the curvature of a "
            f"slope rather than a peak and the posterior is likely far too "
            f"narrow. Raise n_map_steps (10x converged every affected fit in "
            f"issue #1537), or pass an already-converged init_from. "
            f"Set stationarity_tol to silence this.",
            LaplaceNotAtModeWarning,
            stacklevel=2,
            grad_norm=grad_norm,
            newton_decrement=decrement,
            stationarity_tol=stationarity_tol,
        )

    if verbose:
        print(f"  Newton decrement: {decrement:.4g} nats (|grad| {grad_norm:.4g})")

    # Draw samples from N(theta_MAP, H^{-1})
    samples_flat = jax.random.multivariate_normal(key, theta_flat, cov, shape=(n_samples,))

    # Laplace log-evidence estimate
    loss_at_map = float(loss_fn(map_params_unbounded, data_args))
    log_det_h = jnp.sum(jnp.log(eigenvalues_clipped))
    log_evidence = -loss_at_map + 0.5 * n_dim * jnp.log(2.0 * jnp.pi) - 0.5 * log_det_h

    if verbose:
        print(f"  Loss at MAP: {loss_at_map:.2f}")
        print(f"  Laplace log-evidence: {float(log_evidence):.2f}")
        print(f"  Condition number: {float(eigenvalues_clipped[-1] / eigenvalues_clipped[0]):.1e}")

    # Unravel and convert to physical space
    samples_phys = _vmap_samples_to_physical(samples_flat, unravel_fn, to_physical_fn)
    best_params = _mean_params(samples_phys)

    wall_time = time.time() - t0
    if verbose:
        print(f"  Laplace complete in {wall_time:.1f}s")

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Laplace",
        wall_time_s=wall_time,
        diagnostics={
            "n_samples": n_samples,
            "log_evidence": float(log_evidence),
            "loss_at_map": loss_at_map,
            "eigenvalues": eigenvalues_clipped,
            "n_clipped_eigenvalues": n_clipped,
            "condition_number": float(eigenvalues_clipped[-1] / eigenvalues_clipped[0]),
            "newton_decrement": decrement,
            "grad_norm_at_expansion": grad_norm,
        },
        loss_history=None,
        _model=model,
    )
