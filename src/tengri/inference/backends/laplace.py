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

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical


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
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        Samples from the Gaussian approximation, with Laplace log-evidence.
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
            grad_fn, theta_flat, unravel_fn, data_args,
        )
    else:
        def loss_flat(x):
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
        },
        loss_history=None,
        _model=model,
    )
