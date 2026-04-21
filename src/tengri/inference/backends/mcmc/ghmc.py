"""Generalized HMC via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc.common``.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _get_cached_adaptation,
    _get_flat_logdensity,
    _ghmc_burnin_scan,
    _ghmc_sample_scan,
    _set_cached_adaptation,
)

logger = logging.getLogger(__name__)


def run_ghmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    alpha=0.8,
    delta=0.65,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    verbose=True,
):
    """Generalized HMC (GHMC) sampling via BlackJAX.

    GHMC uses partial momentum refreshment controlled by ``alpha``:
    at each step, the momentum is mixed with fresh noise as
    ``p_new = alpha * p_old + sqrt(1-alpha²) * noise``. This creates
    persistent chains that remember their direction, improving mixing
    in elongated posteriors like the age-dust-metallicity banana.

    Parameters
    ----------
    n_warmup : int
        Warmup/adaptation steps.
    n_burnin : int
        Post-warmup burn-in steps (discarded).
    n_samples : int
        Posterior samples to collect.
    alpha : float
        Momentum persistence parameter (0-1). 0 = full refresh (standard HMC),
        1 = no refresh (deterministic). Default 0.8 gives good mixing for
        correlated posteriors.
    delta : float
        Step size scaling in the GHMC proposal. Default 0.65.
    target_accept_rate : float
        Target acceptance rate for warmup adaptation.
    dense_mass_matrix : bool
        Use dense mass matrix. Set False for D>30.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )
    n_dim = len(init_flat)

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "GHMC: %d parameters, %d warmup%s, %d samples, alpha=%.1f, delta=%.2f",
            n_dim, n_warmup, burnin_msg, n_samples, alpha, delta
        )

    t0 = time.time()

    # GHMC's momentum generator treats momentum_inverse_scale as a
    # diagonal vector, so we must use diagonal mass matrix regardless
    # of the dense_mass_matrix flag.
    adapt_key = ("hmc", True)  # always diagonal for GHMC
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    if cached is not None:
        parameters = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )
    else:
        key, warmup_key = jax.random.split(key)
        warmup_hmc = blackjax.window_adaptation(
            blackjax.hmc,
            ld_1arg,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=target_accept_rate,
            num_integration_steps=10,
        )
        (_, parameters), _ = warmup_hmc.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )

    step_size = parameters["step_size"]
    momentum_inv_scale = parameters["inverse_mass_matrix"]

    key, ghmc_init_key = jax.random.split(key)
    state = blackjax.mcmc.ghmc.init(init_flat, ghmc_init_key, ld_1arg)

    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _ghmc_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            momentum_inv_scale,
            alpha,
            delta,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _ghmc_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        momentum_inv_scale,
        alpha,
        delta,
        data_args,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  GHMC complete in %.1fs. Divergences: %d/%d",
            wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="GHMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "alpha": alpha,
            "delta": delta,
            "n_divergent": n_divergent,
            "step_size": float(step_size),
        },
        loss_history=None,
        _model=fitter.model,
    )
