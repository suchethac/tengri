# SPDX-License-Identifier: BSD-3-Clause
"""NUTS (No-U-Turn Sampler) via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.
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
    _nuts_chain_scan,
    _nuts_full_scan,
    _set_cached_adaptation,
    _vmap_chains,
)
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)


def _maybe_warn_high_memory_nuts(n_dim: int, dense_mass_matrix: bool, spec) -> None:
    """Warn before NUTS warmup OOMs at D >= 8 with dense mass matrix (#319).

    The trace graph for full mass-matrix adaptation grows quadratically
    in D and is amplified by SFH variants that publish many per-sample
    derived quantities (``mean_sfh_type="dense_basis"`` is the
    documented worst case — peak 22.78 GB on a D=8 photometry fit).
    Small D <= 7 fits peak at 3-6 GB and are fine.

    Pulled out of :func:`run_nuts` so the heuristic is unit-testable
    without spinning up a full NUTS warmup.
    """
    if not (dense_mass_matrix and n_dim >= 8):
        return
    if getattr(spec, "stochastic", False):
        # Stochastic-SFH fits get a separate, more aggressive warning
        # higher up in run_nuts already.
        return
    heavy_sfh_hint = ""
    mean_sfh_type = getattr(spec, "mean_sfh_type", None)
    if mean_sfh_type is not None:
        types_iter = mean_sfh_type if isinstance(mean_sfh_type, list) else [mean_sfh_type]
        if any("dense_basis" in str(t) for t in types_iter):
            heavy_sfh_hint = (
                " (your mean_sfh_type includes 'dense_basis', which "
                "amplifies this — peak was 22.78 GB on a D=8 fit in "
                "the original report)"
            )
    warnings.warn(
        f"NUTS warmup with dense_mass_matrix=True at D={n_dim} can "
        f"peak at 20+ GB of RAM{heavy_sfh_hint}. If you're on a "
        "32 GB machine and the fit is OOM-ing, pass "
        "`dense_mass_matrix=False` (diagonal mass matrix; small "
        "convergence cost) or switch to `method='mcmc_hmc'` "
        "(less memory-intensive at warmup). See issue #319.",
        stacklevel=3,
    )


def run_nuts(
    context,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_chains=1,
    target_accept_rate=0.85,
    max_num_doublings=10,
    dense_mass_matrix=True,
    pathfinder_warmstart=False,
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
        Posterior samples per chain to collect. 1000 gives convergence
        for most SED fitting scenarios at D≤10. Increase if
        ``check_convergence()`` reports unconverged parameters.
    n_chains : int, default 1
        Number of independent NUTS chains to run in parallel via
        ``jax.vmap`` over chain seeds. Each chain shares the cached
        warmup adaptation (so this is only honoured on the second
        ``run_nuts(...)`` call against the same model — the first call
        populates the cache). Final posterior has ``n_chains * n_samples``
        total samples. Wall ≈ one chain's worth up to the arithmetic
        ceiling (CPU SIMD; GPU/TPU scales further). Initial chain
        positions are MAP + small Gaussian jitter so the chains
        explore independent neighborhoods.
    target_accept_rate : float
        Target acceptance rate for step size adaptation. 0.85 is
        slightly more conservative than the Stan default (0.8),
        reducing divergences in the SED degeneracy banana. Range
        0.7-0.95; higher = smaller steps = fewer divergences but
        slower mixing.
    max_num_doublings : int, default 10
        Maximum tree depth for NUTS trajectory (2^max_num_doublings
        leapfrog steps per sample). Default 10 follows BlackJAX/Stan
        convention. Lower to 7 (2^7=128 leapfrogs/step) if you observe
        a chain running far longer than expected and want to bound the
        worst-case per-sample cost — at the price of higher
        autocorrelation on heavy-tailed or strongly-curved posteriors.
        See ``bench/reports/2026-05-06_compile_vs_sampling_breakdown.md``
        for the prior measurement that found 10→8 gave no benefit on
        nb06's well-conditioned spec posterior; the cap matters only
        when NUTS is hitting deep trees, which is workload-specific.
        Compile cost scales with this knob but is typically <3s at
        warm cache (see docs/inference/compilation_diagnostics.md).
    dense_mass_matrix : bool
        Use a dense (full) mass matrix instead of diagonal. Captures
        parameter correlations (e.g. age-dust-metallicity) and
        dramatically reduces divergences. Default True. Set False
        for D>20 where the dense matrix becomes expensive.
    pathfinder_warmstart : bool, default False
        Use ``blackjax.pathfinder_adaptation`` (L-BFGS mode-finding +
        Hessian-derived inverse mass matrix + short step-size refinement)
        instead of the default window adaptation. Expected to be 3-10x
        faster warmup on *high-dimensional* problems (D>~30). When
        enabled, ``dense_mass_matrix`` is ignored — Pathfinder always
        returns a full inverse-covariance matrix.

        **At low D (<~20), window adaptation is faster and produces
        better posterior geometry.** See
        ``bench/reports/2026-04-22_pathfinder_vs_window_nuts.md``.
        If you enable this, keep ``n_warmup >= 300`` — reducing it
        blindly gives a poorly-conditioned mass matrix that silently
        saturates NUTS tree depth and slows sampling 10-20x.

        References:

        - Zhang et al. 2022, "Pathfinder: Parallel quasi-Newton variational
          inference", JMLR 23, 306, arXiv:2108.03782.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for NUTS: pip install blackjax") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    # ``_shared.py`` helpers still take a Fitter; reach through the
    # context until they migrate. Normalize first so callers can pass
    # either a Fitter or an InferenceContext (matches HMC/raytrace).
    context = InferenceContext.from_target(context)
    fitter = context.fitter

    # Warn about high dimensionality
    if context.spec.stochastic:
        n_total = context.spec.n_free + context.spec.n_grid
        warnings.warn(
            f"Stochastic SFH with NUTS: sampling {n_total} dimensions "
            f"({context.spec.n_grid} psd_xi + {context.spec.n_free} physical). "
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
        _maybe_warn_high_memory_nuts(n_dim, dense_mass_matrix, context.spec)
        # Auto-adjust warnings based on dimensionality
        if n_dim > 20 and not context.spec.stochastic:
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

    adapt_key = ("nuts", not use_dense, bool(pathfinder_warmstart))
    cached = _get_cached_adaptation(fitter, adapt_key)

    if cached is not None:
        parameters = cached

        def ld_1arg(pos):
            return log_posterior_flat_2arg(pos, data_args)

        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
        key, chain_key = jax.random.split(key)
        if n_chains > 1:

            def _init(p):
                return blackjax.mcmc.nuts.init(p, ld_1arg)

            def _scan(s, ks):
                return _nuts_chain_scan(
                    s,
                    ks,
                    log_posterior_flat_2arg,
                    data_args,
                    parameters["step_size"],
                    parameters["inverse_mass_matrix"],
                    max_num_doublings,
                )

            with compile_timer(
                "nuts_chain_scan_vmap", fitter.compile_signature(), method="mcmc_nuts"
            ):
                positions, divergent = _vmap_chains(
                    _init,
                    _scan,
                    init_flat=init_flat,
                    chain_key=chain_key,
                    n_chains=n_chains,
                    n_iter=n_burnin + n_samples,
                    n_burnin=n_burnin,
                )
                jax.block_until_ready(positions)
            _multichain_burnin_done = True
        else:
            state = blackjax.mcmc.nuts.init(init_flat, ld_1arg)
            chain_keys = jax.random.split(chain_key, n_burnin + n_samples)
            with compile_timer("nuts_chain_scan", fitter.compile_signature(), method="mcmc_nuts"):
                positions, divergent = _nuts_chain_scan(
                    state,
                    chain_keys,
                    log_posterior_flat_2arg,
                    data_args,
                    parameters["step_size"],
                    parameters["inverse_mass_matrix"],
                    max_num_doublings,
                )
                jax.block_until_ready(positions)
            _multichain_burnin_done = False
    else:
        _multichain_burnin_done = False
        key, warmup_key = jax.random.split(key)
        key, chain_key = jax.random.split(key)
        chain_keys = jax.random.split(chain_key, n_burnin + n_samples)
        with compile_timer("nuts_full_scan", fitter.compile_signature(), method="mcmc_nuts"):
            positions, divergent, step_size, inv_mass_matrix = _nuts_full_scan(
                init_flat,
                warmup_key,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                max_num_doublings,
                use_dense,
                target_accept_rate,
                bool(pathfinder_warmstart),
            )
            jax.block_until_ready(positions)
        parameters = {"step_size": step_size, "inverse_mass_matrix": inv_mass_matrix}
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup + chain complete (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(step_size),
            )

    # Burnin discard happens Python-side (not inside JIT) so changing
    # n_burnin doesn't trigger a recompile when n_burnin + n_samples is
    # held constant. The multichain branch already discarded per-chain
    # before flattening; skip the global slice there.
    if n_burnin > 0 and not _multichain_burnin_done:
        positions = positions[n_burnin:]
        divergent = divergent[n_burnin:]
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
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
            "n_chains": n_chains,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
            "warmup": "pathfinder" if pathfinder_warmstart else "window",
        },
        loss_history=None,
        _model=context.model,
    )
