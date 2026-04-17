"""Nested Slice Sampling for Bayesian evidence (log Z).

Extracted from fitter.py.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp


def run_nss(
    fitter,
    *,
    key,
    init_from=None,
    n_live=500,
    num_delete=50,
    num_inner_steps=None,
    log_evidence_tol=-3.0,
    max_iterations=10000,
    n_posterior_samples=1000,
    max_steps=10,
    max_shrinkage=100,
    verbose=True,
):
    """Nested Slice Sampling for Bayesian evidence computation.

    Uses Hit-and-Run Slice Sampling (HRSS) as the inner kernel.
    Based on Yallup, Kroupa & Handley (2026, arXiv:2601.23252).

    Restricted to parametric (non-stochastic) SFH models where D ≲ 30.

    Parameters
    ----------
    n_live : int
        Number of live points.
    num_delete : int
        Points to replace per iteration.
    num_inner_steps : int or None
        HRSS walk length per replacement. Defaults to D.
    log_evidence_tol : float
        Terminate when log(Z_remaining) - log(Z_accumulated) < this.
    max_iterations : int
        Safety limit on iterations.
    n_posterior_samples : int
        Number of posterior samples to draw after convergence.
    max_steps : int
        Maximum stepping-out steps in slice sampling.
    max_shrinkage : int
        Maximum shrinking steps in slice sampling.
    verbose : bool
        Print progress.
    """
    from tengri.inference.backends.nested.nss import as_top_level_api
    from tengri.inference.backends.nested.utils import ess as ns_ess, finalise, sample as ns_sample
    from tengri.inference.posterior import Posterior

    # Guard stochastic models
    if fitter.spec.stochastic:
        raise ValueError(
            "NSS not supported for stochastic SFH models (D~137). "
            "Use 'vi' or 'mcmc_raytrace' instead."
        )

    logprior_fn = fitter._build_logprior_fn()
    loglikelihood_fn = fitter._get_or_build_loglikelihood_fn()
    data_args = fitter._data_args

    D = len(fitter._free_names)
    if num_inner_steps is None:
        num_inner_steps = D

    if verbose:
        print(
            f"NSS: {D} parameters, {n_live} live points, "
            f"{num_delete} deletions/iter, {num_inner_steps} HRSS steps"
        )

    # Build NSS algorithm — bind data_args into the likelihood
    def _nss_loglik(free_params):
        return loglikelihood_fn(free_params, data_args)

    algo = as_top_level_api(
        logprior_fn,
        _nss_loglik,
        num_inner_steps,
        num_delete=num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    # Initialize live points from prior
    key, init_key = jax.random.split(key)
    all_samples = fitter.spec.sample_batch(init_key, n_live)
    particles = {name: all_samples[name] for name in fitter._free_names}

    live = algo.init(particles)
    step = jax.jit(algo.step)

    dead_points = []
    n_iter = 0
    t0 = time.time()

    while True:
        key, subkey = jax.random.split(key)
        live, dead = step(subkey, live)
        dead_points.append(dead)
        n_iter += 1

        # Check termination
        logZ_est = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))
        remaining = float(live.integrator.logZ_live - live.integrator.logZ)

        if verbose and n_iter % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  NSS iter {n_iter}: log Z ≈ {logZ_est:.2f}, "
                f"n_dead={n_iter * num_delete}, "
                f"elapsed={elapsed:.1f}s"
            )

        if remaining < log_evidence_tol:
            break
        if n_iter >= max_iterations:
            if verbose:
                print("  NSS: max iterations reached")
            break

    wall_time = time.time() - t0
    logZ = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))

    # Finalise and extract posterior samples
    ns_run = finalise(live, dead_points)

    key, sample_key = jax.random.split(key)
    resampled = ns_sample(sample_key, ns_run, n_posterior_samples)

    key, ess_key = jax.random.split(key)
    ess_val = float(ns_ess(ess_key, ns_run))

    # Convert to physical param dict
    samples_phys = {}
    for name in fitter._free_names:
        samples_phys[name] = resampled.position[name]
    # Add fixed params (broadcast)
    for name, val in fitter._fixed_values.items():
        samples_phys[name] = jnp.full(n_posterior_samples, val)

    best_params = {k: jnp.median(v, axis=0) for k, v in samples_phys.items()}

    if verbose:
        print(f"  NSS complete in {wall_time:.1f}s. log Z = {logZ:.2f}, ESS = {ess_val:.0f}")

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="NSS (Yallup+2026)",
        wall_time_s=wall_time,
        diagnostics={
            "n_live": n_live,
            "num_delete": num_delete,
            "num_inner_steps": num_inner_steps,
            "n_iterations": n_iter,
            "n_dead": n_iter * num_delete,
            "log_evidence": logZ,
            "log_evidence_err": float(jnp.sqrt(jnp.maximum(ess_val, 1.0)) / n_live),
            "ess": ess_val,
        },
        log_evidence=logZ,
        _model=fitter.model,
    )
