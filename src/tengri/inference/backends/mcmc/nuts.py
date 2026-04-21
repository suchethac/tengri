"""NUTS (No-U-Turn Sampler) via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc.common``.
"""

from __future__ import annotations

import logging
import time
import warnings

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _get_cached_adaptation,
    _get_flat_logdensity,
    _nuts_burnin_scan,
    _nuts_sample_scan,
    _set_cached_adaptation,
)

logger = logging.getLogger(__name__)


def run_nuts(
    fitter,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    target_accept_rate=0.85,
    max_num_doublings=10,
    dense_mass_matrix=True,
    verbose=True,
):
    """NUTS sampling via BlackJAX.

    Default strategy designed for SED fitting posteriors with strong
    degeneracies (age-dust-metallicity, SFR-mass), bounded parameters,
    and D=6-20 free parameters.

    The sampling proceeds in four phases:

    1. **MAP initialization** (automatic if ``init_from`` is None):
       200-step MAP optimization to find the posterior mode. Starts
       warmup near the mode instead of the prior, reducing warmup
       time by ~5x and improving mass matrix quality.
    2. **Warmup** (300 steps): BlackJAX window adaptation tunes
       step size and mass matrix.  Dense mass matrix (default)
       captures parameter correlations — critical for the
       age-dust-metallicity degeneracy.
    3. **Burn-in** (100 steps): post-warmup samples discarded.
       Lets the chain diffuse away from the MAP point estimate
       to the typical set of the posterior.
    4. **Sampling** (1000 steps): posterior samples collected.

    Convergence criterion: N > 5τ for all parameters (Sokal/Behroozi).
    Check with ``result.check_convergence()``.

    Parameters
    ----------
    init_from : str, Posterior, or None
        Initialization strategy. None (default) runs a quick MAP
        for warm-starting the chain. Pass a Posterior to start from
        a previous result.
    n_warmup : int
        Warmup/adaptation steps (tunes step size and mass matrix).
        300 is sufficient for D≤15 with MAP init. Increase for
        high-D or difficult geometries.
    n_burnin : int
        Post-warmup burn-in steps (discarded). Lets the chain forget
        the MAP initialization and reach the typical set. Set to 0
        if init_from is already a Posterior from a converged chain.
    n_samples : int
        Posterior samples to collect. 1000 gives convergence for
        most SED fitting scenarios at D≤10. Increase if
        ``check_convergence()`` reports unconverged parameters.
    target_accept_rate : float
        Target acceptance rate for step size adaptation. 0.85 is
        slightly more conservative than the Stan default (0.8),
        reducing divergences in the SED degeneracy banana. Range
        0.7-0.95; higher = smaller steps = fewer divergences but
        slower mixing.
    max_num_doublings : int
        Maximum tree depth for NUTS trajectory (2^max_num_doublings
        leapfrog steps per sample).
    dense_mass_matrix : bool
        Use a dense (full) mass matrix instead of diagonal. Captures
        parameter correlations (e.g. age-dust-metallicity) and
        dramatically reduces divergences. Default True. Set False
        for D>20 where the dense matrix becomes expensive.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for NUTS: pip install blackjax") from None

    from tengri.inference.posterior import Posterior

    # Warn about high dimensionality
    if fitter.spec.stochastic:
        n_total = fitter.spec.n_free + fitter.spec.n_grid
        warnings.warn(
            f"Stochastic SFH with NUTS: sampling {n_total} dimensions "
            f"({fitter.spec.n_grid} psd_xi + {fitter.spec.n_free} physical). "
            f"This is computationally expensive. "
            f"Recommended: method='vi' (10-100x faster).",
            stacklevel=3,
        )

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    if verbose:
        n_dim = len(init_flat)
        # Auto-adjust warnings based on dimensionality
        if n_dim > 20 and not fitter.spec.stochastic:
            warnings.warn(
                f"NUTS with {n_dim} dimensions may be slow. "
                f"Consider method='vi' or method='mcmc_raytrace' for D>{20}.",
                stacklevel=3,
            )
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "NUTS: %d parameters, %d warmup%s, %d samples, target_accept=%.2f",
            n_dim,
            n_warmup,
            burnin_msg,
            n_samples,
            target_accept_rate,
        )

    t0 = time.time()

    # Auto-select mass matrix type based on dimensionality.
    # Dense: O(D²) per step, captures correlations. Good for D≤30.
    # Diagonal: O(D) per step, sufficient when init_from=Posterior
    #   (VI already decorrelated) or D>30.
    n_dim = len(init_flat)
    use_dense = dense_mass_matrix
    if dense_mass_matrix and n_dim > 30:
        use_dense = False
        if verbose:
            logger.info(
                "  Auto-switching to diagonal mass matrix (D=%d>30). "
                "Dense would be O(D²)=%d per step.",
                n_dim,
                n_dim**2,
            )

    adapt_key = ("nuts", not use_dense)
    cached = _get_cached_adaptation(fitter, adapt_key)
    if cached is not None:
        parameters = cached

        def ld_1arg(pos):
            """Closure binding log-posterior with data arguments for initialization."""
            return log_posterior_flat_2arg(pos, data_args)

        state = blackjax.mcmc.nuts.init(init_flat, ld_1arg)
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
    else:
        key, warmup_key = jax.random.split(key)

        def ld_1arg(pos):
            """Closure binding log-posterior with data arguments for warmup."""
            return log_posterior_flat_2arg(pos, data_args)

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            ld_1arg,
            is_mass_matrix_diagonal=not use_dense,
            target_acceptance_rate=target_accept_rate,
        )
        (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )

    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    # Burn-in via cached scan (discarded)
    if n_burnin > 0:
        key, burnin_key = jax.random.split(key)
        burnin_keys = jax.random.split(burnin_key, n_burnin)
        state = _nuts_burnin_scan(
            state,
            burnin_keys,
            log_posterior_flat_2arg,
            step_size,
            inv_mass_matrix,
            max_num_doublings,
            data_args,
        )
        if verbose:
            logger.info("  Burn-in complete (%d steps discarded)", n_burnin)

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_samples)

    _, (positions, divergent) = _nuts_sample_scan(
        state,
        sample_keys,
        log_posterior_flat_2arg,
        step_size,
        inv_mass_matrix,
        max_num_doublings,
        data_args,
    )
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    if verbose:
        logger.info("  Sampling complete (%d samples)", n_samples)

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, fitter._to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  NUTS complete in %.1fs. Divergences: %d/%d", wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="NUTS (BlackJAX)",
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
