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
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)


def _resolve_dense_mass_matrix(dense_mass_matrix: bool | None, n_dim: int) -> bool:
    """Resolve the ``dense_mass_matrix=None`` auto-policy (#319).

    The auto-policy switches to diagonal at D >= 8 to dodge the
    documented 20+ GB warmup spike on photometry fits with
    ``mean_sfh_type="dense_basis"``. Below D = 8 the dense matrix
    converges faster on tengri's typical age-dust-metallicity
    posteriors and peaks at ~3-6 GB.

    Pulled out of :func:`run_nuts` so the heuristic is unit-testable
    without spinning up a full NUTS warmup.

    Parameters
    ----------
    dense_mass_matrix : bool or None
        ``None`` (auto), ``True`` (force dense), or ``False`` (force diagonal).
    n_dim : int
        Number of free parameters in the model.

    Returns
    -------
    bool
        Effective ``dense_mass_matrix`` value to pass to the warmup
        kernel. Explicit ``True`` / ``False`` round-trip unchanged.
    """
    if dense_mass_matrix is None:
        return n_dim < 8
    return dense_mass_matrix


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
        # The `mcmc_hmc` recommendation is sound only because HMC now shares
        # this same auto-policy. While HMC defaulted to `dense_mass_matrix=True`
        # gated on `n_dim <= 30`, it used a DENSE matrix across the whole D =
        # 8-30 band, so this advice sent an OOM-ing user to the more expensive
        # sampler — measured at 13.47 GB and SIGKILLed at D = 9 (#1413, #1454).
        # With both on diagonal above D = 8, HMC's fixed-length trajectory is
        # genuinely lighter than NUTS's adaptive doubling.
        f"NUTS warmup with dense_mass_matrix=True at D={n_dim} can "
        f"peak at 20+ GB of RAM{heavy_sfh_hint}. If you're on a "
        "32 GB machine and the fit is OOM-ing, pass "
        "`dense_mass_matrix=False` (diagonal mass matrix; small "
        "convergence cost) or switch to `method='mcmc_hmc'` "
        "(fixed-length trajectory; lighter warmup). See issue #319.",
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
    dense_mass_matrix: bool | None = None,
    pathfinder_warmstart=False,
    precondition: bool | float | None = None,
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
        warmup adaptation (so this is only honored on the second
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
        warm cache (see docs/performance/compilation.md).
    dense_mass_matrix : bool or None, optional
        Use a dense (full) mass matrix instead of diagonal. Captures
        parameter correlations (e.g. age-dust-metallicity) and
        dramatically reduces divergences.

        Default ``None`` auto-picks based on dimensionality:

        - **D < 8**: dense (warmup peak ~3-6 GB; correlations matter).
        - **D >= 8**: diagonal (avoids the 20+ GB warmup spike at D=8
          with `dense_basis` SFH reported in issue #319).

        Pass ``True`` or ``False`` explicitly to override. Explicit
        ``True`` at D >= 8 emits a memory warning but is honored.
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

    precondition : bool, float or None, default None
        Sample in metric-whitened coordinates. **Opt-in** (#1397): ``None``
        (default) and ``False`` are off — a NaN MAP init makes the metric
        non-finite and turned working fits into hard errors, so the feature must
        be asked for. ``True`` enables it at
        :data:`~tengri.inference.preconditioning.DEFAULT_WHITENING_STRENGTH`, and
        a float in ``[0, 1]`` sets the whitening strength directly (``1.0`` is
        full whitening, ``0.0`` is off).

        Every tengri parameter is standardized, so the prior contributes exactly
        ``I`` to the metric and everything left is the likelihood's — which on the
        correlated-field posterior spans ``cond(grad^2 H) ~ 1e5``, far beyond what
        any single mass matrix estimated from warmup draws can cover. This builds
        the metric analytically at the initial point instead and samples
        ``H(A zeta)`` with ``A A^T = G^-alpha``, mapping draws back with
        ``xi = A zeta``.

        The map is **linear**, so its Jacobian is constant and the posterior is
        unchanged — only the geometry the integrator sees. Pass ``init_from`` a
        MAP result so the metric is built where the chain will actually be.

        **The strength is not cosmetic.** For a true precision ``H`` and a metric
        ``G = H^gamma``, whitening leaves ``cond = cond(H) ** |1 - alpha*gamma|``,
        so full whitening (``alpha=1``) is worse than doing nothing as soon as
        ``gamma > 2`` and amplifies without bound past that. Measured on
        single-galaxy photometry fits, full whitening ranged from 0.10x to 5.76x
        ESS/s across seeds *of the same model* (#1442). The default strength halves
        the exponent and doubles the tolerated misspecification.

        Costs one dense ``(D, D)`` Hessian up front (``O(D)`` backward passes),
        so it is worthwhile at moderate ``D`` and on stiff posteriors, not on
        easy low-dimensional ones.

        This is the metric NIFTy hands to MGVI/geoVI (``I + J^T N^-1 J``); the
        difference is that NIFTy recomputes it every iteration while a
        Hamiltonian sampler needs one fixed metric for the whole chain.

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

    n_dim = len(init_flat)

    # Metric preconditioning (#1301). Every parameter is standardized, so the prior
    # contributes exactly I to the metric and the rest is the likelihood's — which on
    # every configuration measured spans cond 8.5e4 to 3.1e8, far beyond what a mass
    # matrix estimated from warmup draws can cover. Sample the whitened coordinates
    # instead and map the draws back; the map is linear, so the posterior is unchanged.
    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    log_posterior_flat_2arg, init_flat = problem.logdensity, problem.init_flat
    if problem.enabled and verbose:
        # Report the geometry, not the fact that a function was called. Whether the
        # metric was excellent or useless is the whole difference between a 5.76x
        # speedup and a 0.10x slowdown, and it used to be invisible from the log.
        logger.info(
            "NUTS preconditioning: strength=%.2f, cond %.2e -> %.2e at the initial point",
            problem.strength,
            problem.metric_condition,
            problem.whitened_condition,
        )

    # Resolve auto-policy (default since #319). Explicit True/False
    # from the caller is honored as-is.
    user_passed_explicit = dense_mass_matrix is not None
    dense_mass_matrix = _resolve_dense_mass_matrix(dense_mass_matrix, n_dim)
    if verbose and not user_passed_explicit:
        policy = "dense (D<8)" if dense_mass_matrix else "diagonal (D>=8, #319)"
        logger.info("NUTS auto-mass-matrix: %s", policy)

    if verbose:
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
    # (``n_dim`` already computed above for the #319 auto-policy.)
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

    # ``precondition`` changes the sampled geometry, so a cached step size and mass
    # matrix from the un-preconditioned run must not be reused.
    adapt_key = ("nuts", not use_dense, bool(pathfinder_warmstart), problem.cache_key)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Advance the key identically on both branches. Whether a cached adaptation
    # happens to be present is invisible to the caller, so it must not steer the
    # RNG stream: the warmup split below used to live only in the ``else``, which
    # left the cached path one split behind and gave two identical ``fit`` calls
    # with the same ``key`` different chains. ``warmup_key`` is simply unused
    # when the adaptation is reused.
    key, warmup_key = jax.random.split(key)

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

    # Back out of the whitened coordinates before anything interprets the draws as
    # parameters. ``positions`` is (n_draw, D), so ``zeta @ A^T`` applies ``xi = A zeta``
    # row-wise.
    positions = problem.restore(positions)

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
