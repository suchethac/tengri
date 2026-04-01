"""Laplace approximation: Gaussian posterior from Hessian at MAP.

The cheapest uncertainty estimate available — compute the Hessian of the
loss function H(xi) at the MAP estimate, invert to get a covariance matrix,
then draw samples from N(theta_MAP, H^{-1}).

Also provides a Laplace evidence estimate:
    log Z_laplace = -H(theta_MAP) + (D/2)*log(2*pi) - 0.5*log(det(H))

Works entirely in unbounded parameter space (where the loss is smooth),
then transforms samples to physical space.
"""

import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def run_laplace(
    *,
    key,
    loss_fn,
    data_args,
    map_params_unbounded,
    to_physical_fn,
    model,
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

    # Loss in flat space — bind data_args for cache-friendly compilation
    def loss_flat(x):
        return loss_fn(unravel_fn(x), data_args)

    # Hessian at MAP
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
    loss_at_map = float(loss_flat(theta_flat))
    log_det_h = jnp.sum(jnp.log(eigenvalues_clipped))
    log_evidence = -loss_at_map + 0.5 * n_dim * jnp.log(2.0 * jnp.pi) - 0.5 * log_det_h

    if verbose:
        print(f"  Loss at MAP: {loss_at_map:.2f}")
        print(f"  Laplace log-evidence: {float(log_evidence):.2f}")
        print(f"  Condition number: {float(eigenvalues_clipped[-1] / eigenvalues_clipped[0]):.1e}")

    # Unravel and convert to physical space
    samples_phys = {}
    for i in range(n_samples):
        sample_u = unravel_fn(samples_flat[i])
        sample_p = to_physical_fn(sample_u)
        for k, v in sample_p.items():
            if k not in samples_phys:
                samples_phys[k] = []
            samples_phys[k].append(v)

    samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
    best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

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
