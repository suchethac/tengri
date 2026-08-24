# SPDX-License-Identifier: BSD-3-Clause
"""Elliptical Slice Sampling (ESS) for Gaussian-prior latent models.

Murray, Adams & MacKay (2010). Exact MCMC sampler designed for models
with Gaussian priors on latent variables. The proposal moves along
ellipses defined by the prior, guaranteeing acceptance without any
step-size tuning.

In tengri's unbounded parameter space, all parameters have effective
N(0, I) priors, the psd_xi latent field explicitly, and bounded
physical parameters via the sigmoid transform. ESS with cov=I is
therefore mathematically appropriate for the full parameter vector.

ESS takes only the log-LIKELIHOOD (not the full posterior), because
it handles the Gaussian prior internally via the elliptical proposal.

Requires: ``pip install blackjax``
"""

import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical


def run_elliptical_slice(
    *,
    key,
    loglikelihood_unbounded_fn,
    data_args,
    init_params,
    to_physical_fn,
    model,
    n_samples=2000,
    n_burnin=200,
    verbose=True,
):
    """Run Elliptical Slice Sampling via BlackJAX.

    Parameters
    ----------
    key : PRNGKey
        Random key.
    loglikelihood_unbounded_fn : callable
        Log-likelihood function in unbounded space (no prior terms).
        ``(unbounded param dict, data_args) -> scalar``.
    data_args : dict
        Observed data dict (``data``, ``noise``, etc.).
    init_params : dict
        Initial parameters in unbounded space.
    to_physical_fn : callable
        Converts unbounded param dict to physical space.
    model : Model
        Forward model (stored in Posterior).
    n_samples : int
        Number of posterior samples to collect.
    n_burnin : int
        Burn-in steps to discard.
    verbose : bool
        Print progress.

    Returns
    -------
    Posterior
        Posterior samples from elliptical slice sampling.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError(
            "blackjax required for Elliptical Slice Sampling: pip install blackjax"
        ) from None

    from tengri.inference.posterior import Posterior

    t0 = time.time()

    # Flatten init params
    init_flat, unravel_fn = ravel_pytree(init_params)
    n_dim = len(init_flat)

    if verbose:
        print(
            f"Elliptical Slice Sampling: {n_dim} parameters, "
            f"{n_burnin} burn-in, {n_samples} samples"
        )

    # Log-likelihood in flat space (no prior, ESS handles N(0,I) internally)
    def loglik_flat(position):
        """Evaluate log-likelihood in flat parameter space."""
        return loglikelihood_unbounded_fn(unravel_fn(position), data_args)

    # Build ESS kernel with N(0, I) prior
    ess = blackjax.elliptical_slice(
        loglik_flat,
        mean=jnp.zeros(n_dim),
        cov=jnp.eye(n_dim),
    )
    state = ess.init(init_flat)

    @jax.jit
    def one_step(state, rng_key):
        """Execute one ESS step, returning updated state and info."""
        state, info = ess.step(rng_key, state)
        return state, info

    # Burn-in
    key, burnin_key = jax.random.split(key)
    burnin_keys = jax.random.split(burnin_key, n_burnin)
    for sk in burnin_keys:
        state, _ = one_step(state, sk)

    if verbose:
        t_burnin = time.time() - t0
        print(f"  Burn-in complete ({t_burnin:.1f}s)")

    # Sampling
    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    all_positions = []
    subiter_total = 0

    for i, sk in enumerate(sample_keys):
        state, info = one_step(state, sk)
        all_positions.append(state.position)
        if hasattr(info, "subiter"):
            subiter_total += int(info.subiter)
        if verbose and ((i + 1) % 500 == 0 or i == n_samples - 1):
            print(f"  Sample {i + 1}/{n_samples}")

    positions = jnp.stack(all_positions)

    # Unravel and convert to physical space
    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, to_physical_fn)
    best_params = _mean_params(samples_phys)

    wall_time = time.time() - t0
    mean_subiter = subiter_total / max(n_samples, 1)

    if verbose:
        print(f"  ESS complete in {wall_time:.1f}s (mean subiter: {mean_subiter:.1f})")

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Elliptical Slice (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_samples": n_samples,
            "n_burnin": n_burnin,
            "mean_subiter": mean_subiter,
        },
        loss_history=None,
        _model=model,
    )


def run_elliptical_slice_fitter(context, *, key, init_from=None, **kwargs):
    """Elliptical Slice Sampling via the InferenceContext interface.

    Parameters
    ----------
    context : InferenceContext | Fitter
        Inference context (or Fitter, normalized on entry).
    key : PRNGKey
        Random key.
    init_from : Posterior or None
        Initial parameters. If None, use MAP initialization.
    **kwargs
        Passed to run_elliptical_slice (n_samples, n_burnin, verbose).

    Returns
    -------
    Posterior
        Posterior samples from elliptical slice sampling.
    """
    from tengri.inference.context import InferenceContext

    context = InferenceContext.from_target(context)
    # ``_build_loglikelihood_unbounded_fn`` is a Fitter-internal helper.
    loglik_unbounded_fn = context.fitter._build_loglikelihood_unbounded_fn()
    init_params = context.initial_params(key, init_from=init_from)

    return run_elliptical_slice(
        key=key,
        loglikelihood_unbounded_fn=loglik_unbounded_fn,
        data_args=context.data_args,
        init_params=init_params,
        to_physical_fn=context.to_physical,
        model=context.model,
        **kwargs,
    )
