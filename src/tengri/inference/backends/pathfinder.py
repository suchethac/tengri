# SPDX-License-Identifier: BSD-3-Clause
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

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical


def run_pathfinder(
    *,
    key,
    log_posterior_flat,
    init_flat,
    unravel_fn,
    to_physical_fn,
    model,
    n_samples=2000,
    maxiter=30,
    maxcor=10,
    n_elbo_draws=25,
    verbose=True,
):
    """Run BlackJAX Pathfinder for fast approximate posterior.

    Parameters
    ----------
    key : PRNGKey
        Random key.
    log_posterior_flat : callable
        Log-density on flattened parameter vector, cached via
        ``_get_flat_logdensity()`` for stable JIT identity.
    init_flat : jnp.ndarray
        Initial parameters as a flat 1-D array.
    unravel_fn : callable
        Converts flat array back to parameter dict.
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
    n_elbo_draws : int
        Monte-Carlo draws used to estimate the ELBO at each L-BFGS iterate, which
        is how Pathfinder picks the best Gaussian along the path. Default 25,
        matching Stan's ``num_elbo_draws``. **This is a memory knob, not an
        accuracy knob:** the draws are vmapped through the full forward model, so
        peak memory scales as ``n_elbo_draws * maxiter * <cost of one SED>``.
        BlackJAX's own default is 200, which drove a 7-parameter photometry fit to
        26 GB and OOM-killed the slow test tier.
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        Approximate posterior samples from the best Gaussian along the path.

    Notes
    -----
    ``n_samples`` (posterior draws, cheap -- one Gaussian sample each) and
    ``n_elbo_draws`` (path-selection draws, expensive -- one forward model each)
    are different quantities. Raising ``n_samples`` costs almost nothing; raising
    ``n_elbo_draws`` costs a forward evaluation per draw per iterate.

    References
    ----------
    .. [1] L. Zhang, B. Carpenter, A. Gelman & A. Vehtari, "Pathfinder:
       Parallel quasi-Newton variational inference," Journal of Machine Learning
       Research 23(306), 1-49 (2022). arXiv:2108.03782.
    """
    from tengri.inference.backends.mcmc._shared import _check_blackjax_floor

    _check_blackjax_floor()
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for Pathfinder: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    t0 = time.time()

    n_dim = len(init_flat)

    if verbose:
        print(f"Pathfinder: {n_dim} parameters, maxiter={maxiter}, {n_samples} samples")

    key, approx_key, sample_key = jax.random.split(key, 3)

    # Use the module-level ``approximate`` / ``sample`` functions (they take
    # ``logdensity_fn`` explicitly) rather than the instance form
    # ``blackjax.pathfinder(logdensity).approximate(...)``. blackjax 1.4+ made
    # ``blackjax.pathfinder(logdensity)`` return a ``VIAlgorithm`` that no longer
    # carries ``.approximate`` (AttributeError), whereas the module functions
    # have kept a stable signature across ≥1.3, so this works on old and new
    # blackjax without pinning.
    # ``num_samples`` here is blackjax's ELBO-draw count, NOT the posterior draw
    # count -- it defaults to 200 and each draw is a full forward evaluation.
    state, _info = blackjax.pathfinder.approximate(
        approx_key,
        log_posterior_flat,
        init_flat,
        num_samples=n_elbo_draws,
        maxiter=maxiter,
        maxcor=maxcor,
    )

    if verbose:
        t_approx = time.time() - t0
        print(f"  Approximation complete ({t_approx:.1f}s)")

    samples_flat, log_q = blackjax.pathfinder.sample(sample_key, state, n_samples)

    if verbose:
        print(f"  Drew {n_samples} samples (mean log q = {float(jnp.mean(log_q)):.2f})")

    samples_phys = _vmap_samples_to_physical(samples_flat, unravel_fn, to_physical_fn)
    best_params = _mean_params(samples_phys)

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
