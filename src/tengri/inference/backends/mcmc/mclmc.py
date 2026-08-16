# SPDX-License-Identifier: BSD-3-Clause
"""MCLMC and Adjusted MCLMC via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.

**Requires blackjax >= 1.6.** Earlier versions have fundamentally incompatible
APIs. In 1.5, build_kernel requires logdensity_fn and inverse_mass_matrix but
mclmc_find_L_and_step_size does not. In 1.6+, build_kernel takes only the
integrator and mclmc_find_L_and_step_size takes logdensity_fn as an optional
parameter.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _adjusted_mclmc_sample_scan,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _mclmc_sample_scan,
    _set_cached_adaptation,
    _vmap_chains,
)

logger = logging.getLogger(__name__)


def run_mclmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=500,
    n_samples=2000,
    n_chains=1,
    verbose=True,
):
    """Microcanonical Langevin Monte Carlo (MCLMC) via BlackJAX.

    State-of-the-art gradient-based sampler. Operates on the
    microcanonical manifold (constant energy), producing one
    sample per gradient evaluation with no accept/reject step.
    Often 10-100x more efficient than NUTS for smooth targets.

    Because there is no Metropolis correction, MCLMC is **biased**
    for finite step sizes. Use ``adjusted_mclmc`` for exact sampling.

    Parameters
    ----------
    n_warmup : int
        Warmup steps for tuning L (trajectory length) and step size.
    n_samples : int
        Posterior samples to collect.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    # ``_shared.py`` helpers still take a Fitter; reach through
    # the context until they migrate.
    context = InferenceContext.from_target(context)
    fitter = context.fitter

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    def ld_1arg(pos):
        """Closure binding log-posterior with data arguments."""
        return log_posterior_flat_2arg(pos, data_args)

    n_dim = len(init_flat)

    if verbose:
        logger.info("MCLMC: %d parameters, %d warmup, %d samples", n_dim, n_warmup, n_samples)

    t0 = time.time()

    # In blackjax >= 1.6, build_kernel takes only integrator (not logdensity_fn)
    kernel = blackjax.mcmc.mclmc.build_kernel(
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    cached = _get_cached_adaptation(fitter, "mclmc")

    # Both branches must advance the key identically — cache presence is
    # invisible to the caller and must not steer the RNG stream, or two
    # identical ``fit`` calls with one ``key`` return different chains.
    # ``tune_key`` is unused when the adaptation is reused.
    key, init_key = jax.random.split(key)
    key, tune_key = jax.random.split(key)
    state = blackjax.mcmc.mclmc.init(init_flat, ld_1arg, init_key)

    if cached is not None:
        params = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0,
                float(params.L),
                float(params.step_size),
            )
    else:
        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=ld_1arg,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_key,
            diagonal_preconditioning=True,
        )

        _set_cached_adaptation(fitter, "mclmc", params)

        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0,
                float(params.L),
                float(params.step_size),
            )

    key, sample_key = jax.random.split(key)
    if n_chains > 1:

        def _init(p, init_key):
            return blackjax.mcmc.mclmc.init(p, ld_1arg, init_key)

        def _scan(s, ks):
            _, p = _mclmc_sample_scan(
                s, ks, kernel, params.L, params.step_size, ld_1arg, params.inverse_mass_matrix
            )
            return p

        positions = _vmap_chains(
            _init,
            _scan,
            init_flat=init_flat,
            chain_key=sample_key,
            n_chains=n_chains,
            n_iter=n_samples,
            n_burnin=0,
        )
    else:
        sample_keys = jax.random.split(sample_key, n_samples)
        _, positions = _mclmc_sample_scan(
            state,
            sample_keys,
            kernel,
            params.L,
            params.step_size,
            ld_1arg,
            params.inverse_mass_matrix,
        )

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info("  MCLMC complete in %.1fs (%d samples)", wall_time, n_samples)

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="MCLMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "L": float(params.L),
            "step_size": float(params.step_size),
        },
        loss_history=None,
        _model=context.model,
    )


def run_adjusted_mclmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=500,
    n_samples=2000,
    n_chains=1,
    target_accept_rate=0.65,
    verbose=True,
):
    """Adjusted MCLMC (Metropolis-corrected) via BlackJAX.

    Exact (unbiased) version of MCLMC. Adds a Metropolis correction
    step that guarantees detailed balance, making the chain converge
    to the exact target distribution regardless of step size.

    Slightly less efficient than vanilla MCLMC per step (due to the
    accept/reject overhead), but eliminates the step-size-dependent
    bias.

    Parameters
    ----------
    n_warmup : int
        Warmup steps for tuning step size and trajectory length.
    n_samples : int
        Posterior samples to collect.
    target_accept_rate : float
        Target Metropolis acceptance rate. Default 0.65 balances
        bias correction with mixing efficiency.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    fitter = context.fitter
    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    def ld_1arg(pos):
        """Closure binding log-posterior with data arguments."""
        return log_posterior_flat_2arg(pos, data_args)

    n_dim = len(init_flat)

    if verbose:
        logger.info(
            "Adjusted MCLMC: %d parameters, %d warmup, %d samples, target_accept=%.2f",
            n_dim,
            n_warmup,
            n_samples,
            target_accept_rate,
        )

    t0 = time.time()

    # In blackjax >= 1.6, build_kernel takes only integrator (not logdensity_fn)
    kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    cached = _get_cached_adaptation(fitter, "adjusted_mclmc")

    # Both branches must advance the key identically — see run_mclmc above.
    # ``tune_key`` is unused when the adaptation is reused.
    key, tune_key = jax.random.split(key)
    state = blackjax.mcmc.adjusted_mclmc.init(init_flat, ld_1arg)

    if cached is not None:
        params = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0,
                float(params.L),
                float(params.step_size),
            )
    else:
        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=ld_1arg,
            num_steps=n_warmup,
            state=state,
            rng_key=tune_key,
            target=target_accept_rate,
            diagonal_preconditioning=True,
        )

        _set_cached_adaptation(fitter, "adjusted_mclmc", params)

        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). L=%.4f, step_size=%.4f",
                time.time() - t0,
                float(params.L),
                float(params.step_size),
            )

    L = params.L
    step_size = params.step_size
    inverse_mass_matrix = params.inverse_mass_matrix
    n_integration_steps = jnp.ceil(L / step_size).astype(int)

    key, sample_key = jax.random.split(key)
    if n_chains > 1:

        def _init(p):
            return blackjax.mcmc.adjusted_mclmc.init(p, ld_1arg)

        def _scan(s, ks):
            _, (p, d) = _adjusted_mclmc_sample_scan(
                s, ks, kernel, step_size, n_integration_steps, ld_1arg, inverse_mass_matrix
            )
            return p, d

        positions, divergent = _vmap_chains(
            _init,
            _scan,
            init_flat=init_flat,
            chain_key=sample_key,
            n_chains=n_chains,
            n_iter=n_samples,
            n_burnin=0,
        )
    else:
        sample_keys = jax.random.split(sample_key, n_samples)
        _, (positions, divergent) = _adjusted_mclmc_sample_scan(
            state,
            sample_keys,
            kernel,
            step_size,
            n_integration_steps,
            ld_1arg,
            inverse_mass_matrix,
        )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  Adjusted MCLMC complete in %.1fs. Divergences: %d/%d",
            wall_time,
            n_divergent,
            n_samples,
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Adjusted MCLMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_divergent": n_divergent,
            "L": float(params.L),
            "step_size": float(params.step_size),
            "n_integration_steps": int(n_integration_steps),
        },
        loss_history=None,
        _model=context.model,
    )


# ── Elliptical Slice Sampling ─────────────────────────────────────
