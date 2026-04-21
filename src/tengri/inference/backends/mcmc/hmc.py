"""Hamiltonian Monte Carlo (HMC) sampling via BlackJAX.

Standard HMC with fixed trajectory length. Predictable cost per step
(no tree building), making it faster than NUTS per sample when the
geometry is well-conditioned.
"""

from __future__ import annotations

import logging
import time

import jax

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _get_cached_adaptation,
    _get_flat_logdensity,
    _hmc_burnin_scan,
    _hmc_sample_scan,
    _set_cached_adaptation,
)

logger = logging.getLogger(__name__)


def run_hmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_leapfrog_steps=10,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    verbose=True,
):
    """HMC sampling via BlackJAX.

    Standard Hamiltonian Monte Carlo with fixed trajectory length.
    Predictable cost per step (no tree building), making it faster
    than NUTS per sample when the geometry is well-conditioned.

    Parameters
    ----------
    init_from : str, Posterior, or None
        Initialization strategy. None (default) runs a quick MAP
        for warm-starting the chain. Pass a Posterior to start from
        a previous result.
    n_warmup : int
        Warmup/adaptation steps (tunes step size and mass matrix).
    n_burnin : int
        Post-warmup burn-in steps (discarded).
    n_samples : int
        Posterior samples to collect.
    n_leapfrog_steps : int
        Number of leapfrog integration steps per proposal.
    target_accept_rate : float
        Target acceptance rate for step size adaptation.
    dense_mass_matrix : bool
        Use dense mass matrix. Set False for D>30.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for HMC: pip install blackjax") from None

    from tengri.inference._sample_utils import _maybe_map_init
    from tengri.inference.posterior import Posterior

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )
    n_dim = len(init_flat)

    use_dense = dense_mass_matrix and n_dim <= 30

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "HMC: %d parameters, %d warmup%s, %d samples, %d leapfrog/step",
            n_dim, n_warmup, burnin_msg, n_samples, n_leapfrog_steps
        )

    t0 = time.time()

    adapt_key = ("hmc", not use_dense)
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    if cached is not None:
        parameters = cached
        state = blackjax.mcmc.hmc.init(init_flat, ld_1arg)
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )
    else:
        key, warmup_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
            num_integration_steps=n_leapfrog_steps,
        )
        (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0, float(parameters['step_size'])
            )

    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _hmc_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            inv_mass_matrix,
            n_leapfrog_steps,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _hmc_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        inv_mass_matrix,
        n_leapfrog_steps,
        data_args,
    )
    n_divergent = int(jax.numpy.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  HMC complete in %.1fs. Divergences: %d/%d",
            wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="HMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_leapfrog_steps": n_leapfrog_steps,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=fitter.model,
    )
