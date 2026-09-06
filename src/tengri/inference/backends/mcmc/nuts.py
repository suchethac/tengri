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

from tengri.config.exceptions import NUTSTreeDepthWarning, warn_measured
from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    DEFAULT_MAX_NUM_DOUBLINGS,
    _get_cached_adaptation,
    _get_flat_logdensity,
    _get_nuts_kernel,
    _nuts_chain_scan,
    _nuts_warmup_only,
    _set_cached_adaptation,
    _stabilize_dense_mass_step,
    _vmap_chains,
    final_window_divergence_frac,
    refuse_dead_sampling,
    refuse_dead_warmup,
    sampling_diagnostics,
)
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)

#: Post-burnin fraction of iterations at the tree-depth cap above which a
#: deep cap triggers NUTSTreeDepthWarning. 0.25 keeps the warning out of
#: healthy runs (a well-adapted chain touches its cap on a few percent of
#: iterations) while catching the pathological regime: the measured
#: continuity fit saturated 46% of iterations at cap 10.
_SATURATION_WARN_FRAC = 0.25

#: Caps below this stay silent when saturated: at cap <= 6 a truncated tree
#: costs at most 63 gradient evaluations, and a low cap is typically a
#: deliberate wall-time bound (taken knowingly, at a measured ESS cost).
#: At cap >= 7 each saturated iteration burns >= 127 gradients on a
#: trajectory the U-turn criterion never terminated, the signature of
#: heavy-tailed or strongly-correlated geometry worth surfacing.
_SATURATION_WARN_MIN_CAP = 7


def _tree_depth_stats(expansions, max_num_doublings: int) -> dict:
    """Tree-depth diagnostics from per-iteration trajectory-expansion counts.

    Parameters
    ----------
    expansions : array_like, shape (n_draws,)
        ``NUTSInfo.num_trajectory_expansions`` per post-burnin iteration.
    max_num_doublings : int
        The cap those counts ran under.

    Returns
    -------
    dict
        ``tree_depth_mean`` / ``tree_depth_max`` / ``frac_max_depth`` /
        ``max_num_doublings``, all Python scalars so the diagnostics dict
        survives pickling and repr without device arrays.
    """
    expansions = jnp.asarray(expansions)
    return {
        "max_num_doublings": int(max_num_doublings),
        "tree_depth_mean": float(jnp.mean(expansions)),
        "tree_depth_max": int(jnp.max(expansions)),
        "frac_max_depth": float(jnp.mean(expansions >= max_num_doublings)),
    }


def _warn_if_tree_depth_saturated(stats: dict) -> None:
    """Warn when a deep tree-depth cap is saturating (see NUTSTreeDepthWarning).

    Silent below ``_SATURATION_WARN_MIN_CAP``: a low cap that saturates is
    usually a deliberate wall-time bound doing its job, not a pathology.
    """
    cap = stats["max_num_doublings"]
    frac = stats["frac_max_depth"]
    if cap < _SATURATION_WARN_MIN_CAP or frac < _SATURATION_WARN_FRAC:
        return
    warn_measured(
        f"NUTS hit its tree-depth cap on {frac:.0%} of post-burnin iterations "
        f"at max_num_doublings={cap}; each such iteration paid up to "
        f"{2**cap - 1} gradient evaluations on a trajectory the U-turn "
        f"criterion never terminated. On SED posteriors this is the signature "
        f"of heavy-tailed priors or strong parameter correlations (the "
        f"nonparametric-SFH StudentT ratio priors are the measured case). "
        f"Check the model before the sampler: a nonparametric SFH whose bin "
        f"edges run past the age of the universe at the fit redshift leaves "
        f"bins with no likelihood, sampling a heavy-tailed prior that nothing "
        f"constrains. Fixing that on a D=9 continuity fit at z=1.5 cut the "
        f"wall from 174 s to 69 s on its own. After that, trajectory LENGTH is "
        f"the lever: mcmc_hmc with n_leapfrog_steps=150 measured a median "
        f"min-ESS of 118 over six seeds against 119 for this sampler, and "
        f"never fell below 64 where 80 steps returned 31 and 60 steps 23 on "
        f"their worst seed. "
        f"dense_mass_matrix=True is not a safe default here: it measured 12 "
        f"divergences per run on that fit (77 per 400 draws before the bins "
        f"were fixed) against 2 for the diagonal, so check n_divergent before "
        f"trusting its higher speed. Lowering max_num_doublings bounds the "
        f"wall instead, at a real ESS cost (cap 6 measured min-ESS 5 on the "
        f"same fit); quick looks only. "
        f"See posterior.diagnostics['tree_depth_mean'/'frac_max_depth'].",
        NUTSTreeDepthWarning,
        stacklevel=3,
        frac_max_depth=frac,
        max_num_doublings=cap,
    )


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
    documented worst case, peak 22.78 GB on a D=8 photometry fit).
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
                "amplifies this, peak was 22.78 GB on a D=8 fit in "
                "the original report)"
            )
    warnings.warn(
        # The `mcmc_hmc` recommendation is sound only because HMC now shares
        # this same auto-policy. While HMC defaulted to `dense_mass_matrix=True`
        # gated on `n_dim <= 30`, it used a DENSE matrix across the whole D =
        # 8-30 band, so this advice sent an OOM-ing user to the more expensive
        # sampler, measured at 13.47 GB and SIGKILLed at D = 9 (#1413, #1454).
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
    max_num_doublings=DEFAULT_MAX_NUM_DOUBLINGS,
    warmup_max_num_doublings: int | None = None,
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
       captures parameter correlations, critical for the
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
        ``run_nuts(...)`` call against the same model, the first call
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
    warmup_max_num_doublings : int or None, default None
        Tree-depth cap for the **window adaptation only**, leaving the sampling
        phase on ``max_num_doublings``. ``None`` keeps the two equal, which is
        what every caller gets unless they ask otherwise.

        **Measured and NOT recommended.** The idea is sound and the mechanism is
        real: a capped adaptation cannot complete the deep trajectories a small
        step size implies, so dual averaging settles on a larger step size and
        the sampling phase then runs shallower trees. Measured on ``ctl-dpl``
        (D=8, 600 warmup + 600 draws, sampling cap 10) that produced **6.16x
        fewer gradient evaluations per draw at seed 7** — and **0.69x, i.e. 45 %
        MORE, at seed 8**. The effect does not merely shrink on a second seed,
        it reverses sign, so on the one fixture it has been tried on it is not a
        reliable win (``bench/reports/2026-08-31_fast_nuts.md`` Finding 9).

        Combining it with ``precondition`` is worse still: both knobs enlarge the
        adapted step size and composing them enlarges it too far — max split-R-hat
        1.0228, min ESS 50.4, the only measured configuration that fails the
        R-hat clause outright.

        What DID replicate on that fixture is ``precondition=0.5`` at full
        warmup: 8.16x and 2.34x fewer gradients per draw on the two seeds, and a
        per-seed cost stable to 5 % where the unpreconditioned control swings
        3.65x. Reach for that first.

        Kept because it is a correct implementation of a real degree of freedom
        and because a knob measured and found wanting is more useful documented
        than deleted. Any use of it must carry its R-hat and min-ESS columns:
        ``bench/reports/2026-04-22_pathfinder_vs_window_nuts.md`` records an 18x
        regression from an under-adapted step size, which is the failure this
        knob courts by construction.
    max_num_doublings : int, default DEFAULT_MAX_NUM_DOUBLINGS (10)
        Maximum tree depth for NUTS trajectory (up to 2^max_num_doublings - 1
        leapfrog steps per sample). Default 10 follows the BlackJAX/Stan
        convention, and survived a deliberate attempt to lower it
        (2026-08-18): on a 19-band continuity fit (D=9, 500+500 draws) cap 6
        cut the wall 118 s → 11 s but collapsed min-ESS 93 → 5, strictly
        worse per effective sample. When a fit saturates this cap (the same
        measurement saw 46% of iterations at depth 10), the geometry, not
        the cap, is the problem, and on a nonparametric SFH part of that
        geometry is self-inflicted: bin edges running past the age of the
        universe leave bins with no likelihood sampling a heavy-tailed prior
        (#1975). Matching the edges to the redshift took the same fit from
        174 s to 69 s. ``mcmc_hmc`` with ``n_leapfrog_steps=150`` then
        measured a median min-ESS of 118 per 400 draws over six seeds
        against 119 here, and never fell below 64.
        ``dense_mass_matrix=True`` is quicker still but measured 12
        divergences per run against 2 for the diagonal, so check
        ``n_divergent`` before trusting it. Lower the cap only to bound
        worst-case wall for a quick look, knowingly paying ESS. Saturation of a cap >= 7 on > 25%
        of iterations warns via
        :class:`~tengri.config.exceptions.NUTSTreeDepthWarning`, and every
        fit reports ``tree_depth_mean`` / ``tree_depth_max`` /
        ``frac_max_depth`` in ``posterior.diagnostics``. The earlier finding
        that 10→8 gave no benefit on nb06's well-conditioned posterior
        (``bench/reports/2026-05-06_compile_vs_sampling_breakdown.md``) is
        the quiet side of the same fact: a chain that never builds deep
        trees does not feel the cap. Compile cost scales with this knob but
        is typically <3s at warm cache (see docs/performance/compilation.md).
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
        enabled, ``dense_mass_matrix`` is ignored, Pathfinder always
        returns a full inverse-covariance matrix.

        **At low D (<~20), window adaptation is faster and produces
        better posterior geometry.** See
        ``bench/reports/2026-04-22_pathfinder_vs_window_nuts.md``.
        If you enable this, keep ``n_warmup >= 300``, reducing it
        blindly gives a poorly-conditioned mass matrix that silently
        saturates NUTS tree depth and slows sampling 10-20x.

        References:

        - Zhang et al. 2022, "Pathfinder: Parallel quasi-Newton variational
          inference", JMLR 23, 306, arXiv:2108.03782.

    precondition : bool, float or None, default None
        Sample in metric-whitened coordinates. **Opt-in** (#1397): ``None``
        (default) and ``False`` are off, a NaN MAP init makes the metric
        non-finite and turned working fits into hard errors, so the feature must
        be asked for. ``True`` enables it at
        :data:`~tengri.inference.preconditioning.DEFAULT_WHITENING_STRENGTH`, and
        a float in ``[0, 1]`` sets the whitening strength directly (``1.0`` is
        full whitening, ``0.0`` is off).

        Every tengri parameter is standardized, so the prior contributes exactly
        ``I`` to the metric and everything left is the likelihood's, which on the
        correlated-field posterior spans ``cond(grad^2 H) ~ 1e5``, far beyond what
        any single mass matrix estimated from warmup draws can cover. This builds
        the metric analytically at the initial point instead and samples
        ``H(A zeta)`` with ``A A^T = G^-alpha``, mapping draws back with
        ``xi = A zeta``.

        The map is **linear**, so its Jacobian is constant and the posterior is
        unchanged, only the geometry the integrator sees. Pass ``init_from`` a
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
    # contributes exactly I to the metric and the rest is the likelihood's, which on
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
    # n_warmup and target_accept_rate belong in the key: they *produce* the
    # adaptation, so leaving them out makes both knobs silently inert on a model
    # that already holds an entry. Grouped into one element and kept on a single
    # line because two tests read this statement as text -- the namespace guard
    # in test_preconditioning.py matches per line, and #1454 matches up to the
    # first ``)``.
    # ``max_num_doublings`` is in the key because it now reaches the *warmup*
    # kernel too (Phase 3): a step size dual-averaged under a cap of 2 is not the
    # one a cap of 10 would have found, so reusing a cached adaptation across
    # caps would silently sample at the wrong step size. Leaving a knob out of
    # this tuple is what makes it inert rather than wrong -- see
    # ``_adaptation_cache_key``.
    # Warmup may run under a SHALLOWER cap than sampling. The two halves are not
    # symmetric: during warmup the step size has not converged, so trees are at
    # their deepest and most heterogeneous exactly where they are least
    # informative -- one vmapped 100-step adaptation over 64 lanes at D = 3 cost
    # 1272 s, ~12.7 s per step, for a three-parameter model
    # (bench/reports/2026-08-31_fast_nuts.md Finding 7). Capping the adaptation
    # harder than the sampling is therefore a knob that did not exist: #2101 made
    # ``max_num_doublings`` reach warmup at all, and it reaches it with the
    # sampling value. ``None`` keeps them equal, so this is a no-op unless asked
    # for.
    #
    # It is not free, and the failure mode has a name: an adaptation that cannot
    # complete a trajectory returns a step size tuned for a truncated one.
    # ``bench/reports/2026-04-22_pathfinder_vs_window_nuts.md`` records an 18x
    # regression from an under-adapted step size, so any use of this must carry
    # its R-hat and ESS columns.
    warmup_max_doublings = int(
        max_num_doublings if warmup_max_num_doublings is None else warmup_max_num_doublings
    )
    tuning = (
        int(n_warmup),
        float(target_accept_rate),
        int(max_num_doublings),
        warmup_max_doublings,
    )
    adapt_key = ("nuts", not use_dense, bool(pathfinder_warmstart), tuning, problem.cache_key)
    cached = _get_cached_adaptation(fitter, adapt_key)

    # Advance the key identically on both branches. Whether a cached adaptation
    # happens to be present is invisible to the caller, so it must not steer the
    # RNG stream: the warmup split below used to live only in the ``else``, which
    # left the cached path one split behind and gave two identical ``fit`` calls
    # with the same ``key`` different chains. ``warmup_key`` is simply unused
    # when the adaptation is reused.
    key, warmup_key = jax.random.split(key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    # ── Warmup: adapt (step_size, inverse_mass_matrix) once, then cache. ──
    # Split from sampling so a fresh call and a cached one end in the SAME
    # sampling scan. While the two were fused, a first fit ran warmup+sampling
    # inside `_nuts_full_scan` and every later fit on the same model ran a
    # sampling-only scan against the cached parameters, structurally different
    # computations, so one pinned `key` returned two different posteriors. HMC
    # had already been split this way and was reproducible; NUTS had not.
    dense_mass_backoffs = 0
    if cached is not None:
        parameters = cached
        # A reused adaptation was tuned in an earlier call, so this fit measured no
        # warmup divergences of its own. The diagnostics key is then ABSENT rather
        # than None: Posterior.save() has no HDF5 representation for None and would
        # warn about a skipped entry on every warm fit (#2088). Presence of the key
        # means "measured in this call".
        warmup_record: dict = {}
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
    else:
        with compile_timer("nuts_warmup", fitter.compile_signature(), method="mcmc_nuts"):
            step_size, inv_mass_matrix, warmup_divergent = _nuts_warmup_only(
                init_flat,
                warmup_key,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                use_dense,
                target_accept_rate,
                bool(pathfinder_warmstart),
                warmup_max_doublings,
            )
            jax.block_until_ready(step_size)

        # Post-adaptation step size stability probe for dense mass matrix (#1999)

        if use_dense:
            import blackjax

            initial_state = blackjax.nuts.init(
                init_flat, lambda p: log_posterior_flat_2arg(p, data_args)
            )
            kernel = _get_nuts_kernel()
            step_size, backoff_count = _stabilize_dense_mass_step(
                kernel,
                initial_state,
                log_posterior_flat_2arg,
                data_args,
                float(step_size),
                inv_mass_matrix,
                warmup_max_doublings,
                sampler_name="NUTS",
            )
            step_size = jnp.asarray(step_size)
            dense_mass_backoffs = int(backoff_count)
            if verbose and backoff_count > 0:
                logger.info(
                    f"  Dense-mass probe: step size backoff applied ({backoff_count} halving(s))"
                )

        # Refuse before caching and before the sampling scan compiles (#2088).
        warmup_divergence_frac = final_window_divergence_frac(warmup_divergent, n_warmup)
        refuse_dead_warmup(
            warmup_divergence_frac,
            sampler="NUTS",
            step_size=float(step_size),
            n_warmup=n_warmup,
            n_samples=n_samples,
        )
        warmup_record = (
            {}
            if warmup_divergence_frac is None
            else {"warmup_divergence_frac": warmup_divergence_frac}
        )
        parameters = {"step_size": step_size, "inverse_mass_matrix": inv_mass_matrix}
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            if warmup_divergence_frac is None:
                logger.info(
                    "  Warmup complete (%.1fs). Step size: %.4f",
                    time.time() - t0,
                    float(step_size),
                )
            else:
                logger.info(
                    "  Warmup complete (%.1fs). Step size: %.4f. "
                    "Divergent in the final warmup window: %.0f%%",
                    time.time() - t0,
                    float(step_size),
                    100.0 * warmup_divergence_frac,
                )

    # ── Sampling: one path, whether the adaptation was just tuned or reused. ──
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

        with compile_timer("nuts_chain_scan_vmap", fitter.compile_signature(), method="mcmc_nuts"):
            positions, divergent, expansions = _vmap_chains(
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
            positions, divergent, expansions = _nuts_chain_scan(
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

    # Burnin discard happens Python-side (not inside JIT) so changing
    # n_burnin doesn't trigger a recompile when n_burnin + n_samples is
    # held constant. The multichain branch already discarded per-chain
    # before flattening; skip the global slice there.
    if n_burnin > 0 and not _multichain_burnin_done:
        positions = positions[n_burnin:]
        divergent = divergent[n_burnin:]
        expansions = expansions[n_burnin:]
    n_divergent = int(jnp.sum(divergent))
    depth_stats = _tree_depth_stats(expansions, max_num_doublings)
    _warn_if_tree_depth_saturated(depth_stats)

    # Post-hoc dead-fit detection: any chain's draws mostly divergent (#2093).
    # Complements the pre-hoc warmup check for fits that collapse after warmup.
    # On failure, evict the cached adaptation so the next fit re-tunes.
    refuse_dead_sampling(
        divergent,
        sampler="NUTS",
        n_samples=n_samples,
        n_chains=n_chains,
        step_size=float(parameters["step_size"]),
        fitter=fitter,
        method_key=adapt_key,
    )

    wall_time = time.time() - t0

    # Back out of the whitened coordinates before anything interprets the draws as
    # parameters. ``positions`` is (n_draw, D), so ``zeta @ A^T`` applies ``xi = A zeta``
    # row-wise.
    positions = problem.restore(positions)

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    # Compute sampling diagnostics (divergence fraction and unique draw count).
    diag = sampling_diagnostics(positions, divergent)

    n_total = n_samples * n_chains
    if verbose:
        logger.info(
            "  NUTS complete in %.1fs. Divergences: %d/%d (%.1f%%). "
            "Tree depth: mean %.1f, at cap on %.0f%% of iterations (max_num_doublings=%d)",
            wall_time,
            n_divergent,
            n_total,
            100.0 * n_divergent / max(n_total, 1),
            depth_stats["tree_depth_mean"],
            100.0 * depth_stats["frac_max_depth"],
            max_num_doublings,
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
            "dense_mass_step_backoffs": dense_mass_backoffs,
            **warmup_record,
            "step_size": float(parameters["step_size"]),
            "warmup": "pathfinder" if pathfinder_warmstart else "window",
            **diag,
            "warmup_max_num_doublings": warmup_max_doublings,
            **depth_stats,
        },
        loss_history=None,
        _model=context.model,
    )
