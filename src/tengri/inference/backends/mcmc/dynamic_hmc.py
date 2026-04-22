"""Dynamic HMC via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _dynamic_hmc_burnin_scan,
    _dynamic_hmc_sample_scan,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _set_cached_adaptation,
)

logger = logging.getLogger(__name__)


def run_dynamic_hmc(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    target_accept_rate=0.85,
    dense_mass_matrix=True,
    verbose=True,
):
    """Dynamic HMC sampling via BlackJAX.

    HMC with dynamic trajectory length selection — adapts the number
    of leapfrog steps per proposal based on a heuristic that balances
    exploration vs cost. Similar to NUTS but without the binary tree.

    Parameters
    ----------
    n_warmup : int
        Warmup/adaptation steps.
    n_burnin : int
        Post-warmup burn-in steps (discarded).
    n_samples : int
        Posterior samples to collect.
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
        raise ImportError("blackjax required: pip install blackjax") from None

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
            "Dynamic HMC: %d parameters, %d warmup%s, %d samples",
            n_dim,
            n_warmup,
            burnin_msg,
            n_samples,
        )

    t0 = time.time()

    # dynamic_hmc.init needs random_generator_arg, incompatible with
    # window_adaptation. Use HMC warmup to tune step_size/mass matrix,
    # then initialize dynamic_hmc state separately.
    adapt_key = ("hmc", not use_dense)
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        """Closure binding log-posterior with data arguments for adaptation."""
        return log_posterior_flat_2arg(pos, data_args)

    if cached is not None:
        parameters = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
    else:
        key, warmup_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
            num_integration_steps=10,
        )
        (_, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )

    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    key, init_key = jax.random.split(key)
    state = blackjax.mcmc.dynamic_hmc.init(init_flat, ld_1arg, init_key)

    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _dynamic_hmc_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            inv_mass_matrix,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _dynamic_hmc_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        inv_mass_matrix,
        data_args,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  Dynamic HMC complete in %.1fs. Divergences: %d/%d",
            wall_time,
            n_divergent,
            n_samples,
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="Dynamic HMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=fitter.model,
    )
