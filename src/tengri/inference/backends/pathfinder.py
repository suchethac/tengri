"""Pathfinder: fast approximate posterior via quasi-Newton L-BFGS path.

Pathfinder (Zhang et al. 2022) traces the L-BFGS optimization trajectory
and fits a sequence of Gaussian approximations along the path. It picks
the best approximation (by ELBO) and draws samples from it.

~10x faster than NUTS for approximate posteriors, and excellent as a
warm-start initializer for NUTS or Ray Tracing chains.

Requires: ``pip install blackjax``
"""

import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def run_pathfinder(
    *,
    key,
    loss_fn,
    data_args,
    init_params,
    to_physical_fn,
    model,
    n_samples=2000,
    maxiter=30,
    maxcor=10,
    verbose=True,
):
    """Run BlackJAX Pathfinder for fast approximate posterior.

    Parameters
    ----------
    key : PRNGKey
        Random key.
    loss_fn : callable
        Loss function: ``(unbounded param dict, data_args) -> scalar``.
    data_args : dict
        Observed data dict (``data``, ``noise``, etc.).
    init_params : dict
        Initial parameters in unbounded space.
    to_physical_fn : callable
        Converts unbounded param dict to physical space.
    model : Model
        Forward model (stored in Posterior).
    n_samples : int
        Number of posterior samples to draw.
    maxiter : int
        Maximum L-BFGS iterations along the path.
    maxcor : int
        L-BFGS memory (number of past gradients to store).
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        Approximate posterior samples from the best Gaussian along the path.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for Pathfinder: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    t0 = time.time()

    # Flatten init params
    init_flat, unravel_fn = ravel_pytree(init_params)
    n_dim = len(init_flat)

    if verbose:
        print(f"Pathfinder: {n_dim} parameters, maxiter={maxiter}, {n_samples} samples")

    # Log-posterior in flat space — bind data_args for cache-friendly compilation
    def log_posterior_flat(position):
        return -loss_fn(unravel_fn(position), data_args)

    # Run Pathfinder approximation
    key, approx_key, sample_key = jax.random.split(key, 3)

    pathfinder = blackjax.pathfinder(log_posterior_flat)
    state, _info = pathfinder.approximate(
        approx_key,
        init_flat,
        maxiter=maxiter,
        maxcor=maxcor,
    )

    if verbose:
        t_approx = time.time() - t0
        print(f"  Approximation complete ({t_approx:.1f}s)")

    # Draw samples from the best Gaussian approximation
    samples_flat, log_q = pathfinder.sample(sample_key, state, n_samples)

    if verbose:
        print(f"  Drew {n_samples} samples (mean log q = {float(jnp.mean(log_q)):.2f})")

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
        print(f"  Pathfinder complete in {wall_time:.1f}s")

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Pathfinder (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_samples": n_samples,
            "maxiter": maxiter,
            "maxcor": maxcor,
            "mean_log_q": float(jnp.mean(log_q)),
        },
        loss_history=None,
        _model=model,
    )
