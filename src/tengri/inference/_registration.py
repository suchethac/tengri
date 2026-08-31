# SPDX-License-Identifier: BSD-3-Clause
"""Backend registry initialization for ``tengri.inference``.

This module is imported for its side effects, every
``@register_backend(...)`` decorator at module load time inserts a
``BackendEntry`` into ``_BACKENDS`` so that ``Fitter.run(method=...)``
can dispatch by name.

It lives in its own module (rather than at the bottom of ``fitter.py``)
to keep that file focused on the ``Fitter`` class itself. Mirrors the
``forward/_kernels`` layout (ADR-0004): adapter registration is a
package-level concern, not an orchestrator-level one.

Adding a new backend
--------------------
1. Implement ``def run_X(context, *, key, init_from=None, ...)`` in
   ``backends/X.py``. The context is an :class:`InferenceContext`;
   read forward-model state through its accessors.
2. Add an entry below: ``register_backend("X", tier="experimental",
   short_doc=..., legacy_fitter=False)(run_X)``.
3. Add a conformance smoke in ``tests/unit/inference/`` (the
   parametrized suite in ``test_backend_conformance.py`` will pick it
   up automatically).

See ADR-0010 for the full Protocol contract.
"""

from __future__ import annotations

from tengri.inference._backend_registry import register_backend
from tengri.inference.backends.evidence import run_nss as _ctx_run_nss
from tengri.inference.backends.map_dispatch import (
    run_laplace as _ctx_run_laplace,
    run_map as _ctx_run_map,
    run_pathfinder as _ctx_run_pathfinder,
)
from tengri.inference.backends.mcmc import (
    run_adjusted_mclmc as _ctx_run_adjusted_mclmc,
    run_barker as _ctx_run_barker,
    run_chees as _ctx_run_chees,
    run_dynamic_hmc as _ctx_run_dynamic_hmc,
    run_ghmc as _ctx_run_ghmc,
    run_hmc as _ctx_run_hmc,
    run_hmc_is as _ctx_run_hmc_is,
    run_hmc_low_rank as _ctx_run_hmc_low_rank,
    run_mala as _ctx_run_mala,
    run_mclmc as _ctx_run_mclmc,
    run_nuts as _ctx_run_nuts,
    run_raytrace as _ctx_run_raytrace,
    run_smc as _ctx_run_smc,
)
from tengri.inference.backends.mcmc.elliptical_slice import (
    run_elliptical_slice_fitter as _ctx_run_elliptical_slice,
)
from tengri.inference.backends.vi.gaussian import (
    run_gaussian_vi_fitter as _ctx_run_gaussian_vi,
)
from tengri.inference.backends.vi.native import run_native_vi as _ctx_run_native_vi
from tengri.inference.backends.vi.nifty import (
    run_nifty_fast_vi as _ctx_run_nifty_fast_vi,
    run_nifty_vi as _ctx_run_nifty_vi,
)


def _mcmc_auto_pick(context, *, key, init_from=None, precondition=None, **kw):
    """``mcmc`` auto-dispatcher: NUTS for low-D, raytrace for high-D.

    Threshold is looked up at call time (not import time) so this
    module has no import dependency on ``fitter.py``, keeps the
    package import graph one-way and lets ``inference/__init__.py``
    rely on plain alphabetical import ordering.

    ``precondition`` is named explicitly rather than left in ``**kw`` because which
    branch runs decides whether it can be honored, and the two branches disagree.
    Which one *is* capable comes from the registry, not from a name written here.
    """
    from tengri.inference._backend_registry import check_capabilities, get_backend
    from tengri.inference.fitter import _MCMC_AUTO_D_THRESHOLD

    if context.spec.n_latent <= _MCMC_AUTO_D_THRESHOLD:
        return _ctx_run_nuts(
            context, key=key, init_from=init_from, precondition=precondition, **kw
        )

    # High-D branch. Ray tracing is not a Hamiltonian sampler, so today it has no
    # integrator metric to whiten, but that is the registry's fact to state, not
    # this function's. Refuse an explicit request rather than drop it silently; a
    # bare ``precondition=None`` is the auto-policy and resolves to off.
    selected = get_backend("mcmc_raytrace")
    check_capabilities(selected, {"precondition": precondition})
    if selected.accepts_precondition:
        kw["precondition"] = precondition
    return _ctx_run_raytrace(context, key=key, init_from=init_from, **kw)


def _run_fullrank_vi(context, *, key, init_from=None, precondition=None, **kw):
    """``vi_fullrank``: Gaussian VI with a Cholesky covariance factor.

    ``precondition`` is named here rather than left in ``**kw`` because
    ``test_preconditioning_capability`` reads the *registered* runner's signature:
    a backend declaring ``accepts_precondition`` whose entry point swallows the
    kwarg in ``**kwargs`` would forward it to something that raises.
    """
    return _ctx_run_gaussian_vi(
        context, key=key, init_from=init_from, precondition=precondition, family="fullrank", **kw
    )


def _run_meanfield_vi(context, *, key, init_from=None, precondition=None, **kw):
    """``vi_meanfield``: Gaussian VI with a diagonal covariance.

    See :func:`_run_fullrank_vi` for why ``precondition`` is spelled out.
    """
    return _ctx_run_gaussian_vi(
        context, key=key, init_from=init_from, precondition=precondition, family="meanfield", **kw
    )


# ── Primary backends ─────────────────────────────────────────────────────
register_backend(
    "map",
    tier="primary",
    short_doc="Adam MAP optimization",
    requires=("optax",),
    legacy_fitter=False,
)(_ctx_run_map)

# NIFTy geoVI/MGVI, sample_mode flags select geoVI vs MGVI.
register_backend(
    "vi",
    tier="primary",
    short_doc=(
        "NIFTy geoVI variational inference (cold ~100s, ~20 GB RSS at D=6-7, "
        "memory-heavy; consider mcmc_hmc for faster turnaround on D<10)"
    ),
    aliases=("vi_nonlinear",),
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "nonlinear_resample", **kw},
    )
)

register_backend(
    "vi_nonlinear_fast",
    tier="primary",
    short_doc="NIFTy geoVI without Python logging",
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_fast_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "nonlinear_resample", **kw},
    )
)

register_backend(
    "vi_linear",
    tier="experimental",
    short_doc="NIFTy MGVI standard with logging",
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "linear_resample", **kw},
    )
)

register_backend(
    "vi_linear_fast",
    tier="experimental",
    short_doc="NIFTy MGVI without Python logging",
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_fast_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "linear_resample", **kw},
    )
)

register_backend(
    "native_vi_nonlinear",
    tier="broken",
    short_doc=(
        "[UNSTABLE] Pure JAX geoVI, segfaults on DPL/dense_basis "
        "photometry mocks (validated 2026-05-22, issue #231). Use 'vi' instead."
    ),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_native_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "geovi", **kw},
    )
)

register_backend(
    "native_vi_linear",
    tier="broken",
    short_doc=(
        "[UNSTABLE] Pure JAX MGVI, segfaults on DPL/dense_basis "
        "photometry mocks (validated 2026-05-22, issue #231). Use 'vi_linear' instead."
    ),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_native_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "linear", **kw},
    )
)

# ``mcmc`` is an auto-dispatcher (NUTS for low-D, raytrace for high-D).
# It picks the concrete runner from ``context.spec.n_free``.
register_backend(
    "mcmc",
    tier="primary",
    short_doc="Auto MCMC: NUTS for low-D, raytrace for high-D",
    requires=("blackjax",),  # NUTS branch needs it; raytrace branch is pure JAX
    legacy_fitter=False,
    accepts_precondition=True,
)(_mcmc_auto_pick)

register_backend(
    "mcmc_nuts",
    tier="primary",
    short_doc=(
        "No-U-Turn Sampler (cold ~90s at D=6 DPL; warmup blows past 5 min on "
        "dense_basis D=7, prefer mcmc_hmc or mcmc_ghmc for dense_basis SFH)"
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_nuts)

register_backend(
    "mcmc_raytrace",
    tier="primary",
    short_doc="Ray-tracing ensemble sampler (high-D)",
    legacy_fitter=False,
)(_ctx_run_raytrace)

# ── Promoted from experimental ──────────────────────────────────────────
# Validated against DPL (D=6) and dense_basis (D=7) photometry mocks
# on 2026-05-22 (issue #231). See docs/dev/benchmarks/2026-05-22_inference_backend_validation.md.
register_backend(
    "laplace",
    tier="primary",
    short_doc="Laplace approximation around the MAP (cold ~5-9s, warm ~1-2s, ~3 GB)",
    legacy_fitter=False,
)(_ctx_run_laplace)

register_backend(
    "mcmc_hmc",
    tier="primary",
    short_doc=(
        "Hamiltonian Monte Carlo (cold ~21s, ~5 GB on D=6-7). "
        "Convergence-validated only with dense_mass_matrix=True, "
        "n_warmup≥1000, n_leapfrog_steps≥20 on D=6 DPL "
        "(R-hat 1.008, ESS 411). Default n_warmup=300 / dense=True "
        "gives R-hat ≫ 1, do not lower the warmup for science."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_hmc)

register_backend(
    "hmc_is",
    tier="experimental",
    short_doc=(
        "HMC posterior + importance-sampled log-evidence (cold ~30s at D=6; "
        "check diagnostics['ess'] and diagnostics['max_weight_frac'] for quality)"
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_hmc_is)

register_backend(
    "mcmc_dynamic_hmc",
    tier="experimental",
    short_doc=(
        "Dynamic HMC, fast (cold ~19s) but chains under-mix at default "
        "settings (R-hat ≈ 1.11-1.25, ESS ≈ 1-30 on D=6-7 mocks, 1000 "
        "warmup + 2000 samples). Needs tuning before science use."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_dynamic_hmc)

register_backend(
    "mcmc_ghmc",
    tier="broken",
    short_doc=(
        "[POOR MIXING] Generalized HMC, fast (cold ~17s) but R-hat ≈ 2.5-3.1 "
        "and ESS ≈ 1 on D=6-8 photometry mocks. The adapter was the suspect "
        "and it has now been replaced: window adaptation → "
        "blackjax.meads_adaptation, the adaptation purpose-built for this "
        "kernel, and it does NOT fix the mixing on the tsnorm posteriors "
        "(R-hat 1.8-14, ESS ≈ 1, and MEADS's step size collapses to ~1e-6). "
        "Do not use for science. Measured six seeds x three notebook models in "
        "bench/reports/2026-08-30_ghmc_meads_adaptation.md; earlier context in "
        "docs/dev/benchmarks/2026-05-22_inference_backend_validation.md."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_ghmc)

register_backend(
    "mcmc_mclmc",
    tier="broken",
    short_doc=(
        "[POOR MIXING] Microcanonical Langevin MC. The earlier quarantine "
        "reason -- 'R-hat ≈ 1.7 and ESS ≈ 1 at 4000 samples' -- was a units "
        "error: an MCLMC draw is one integrator step (2 gradients), a NUTS draw "
        "is a whole trajectory (~50-77 gradients measured here), so 4000 of each "
        "is not the same budget. Tuned and re-measured 2026-08-30: at 40000 "
        "draws it clears max split-R-hat < 1.01 on 6/6 seeds of D=8 nb05 where "
        "shipped NUTS clears 1/6. Still quarantined for a different and real "
        "reason: 2 of those 6 seeds finished at 300x and 170,000x their "
        "energy-error variance target with R-hat still reading 1.0007, and this "
        "sampler is unadjusted, so nothing rejects an over-large step -- the "
        "chains mix to a displaced distribution and R-hat cannot see it. Do not "
        "use for science until that is understood; read energy_var_per_dim, not "
        "R-hat, and see bench/reports/2026-08-30_mclmc_tuning.md. "
        "Requires blackjax >= 1.6."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    # NOT accepts_precondition, though run_mclmc does take `precondition=` and
    # wires it to the same analytic-metric seam NUTS uses. The capability is
    # declared when it has been measured and the tier allows a fit:
    # test_preconditioning_roundtrip parametrises over every backend declaring
    # it and runs a real fit through each, which a tier="broken" backend cannot
    # do. Declaring it here was an unmeasured claim that broke that contract.
)(_ctx_run_mclmc)

# ── Experimental backends ────────────────────────────────────────────────
register_backend(
    "mcmc_chees",
    tier="experimental",
    short_doc=(
        "ChEES-HMC: one trajectory length learned from cross-chain statistics, "
        "so every chain still takes the same number of leapfrogs (lock-step "
        "preserved) while L comes from the posterior rather than from a "
        "hand-set constant. Metropolis-corrected dynamic HMC underneath, with "
        "the metric supplied analytically (precondition=) rather than "
        "estimated from the ensemble. Measured six seeds x four posteriors in "
        "bench/reports/2026-08-30_chees_hmc.md."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_chees)

register_backend(
    "mcmc_smc",
    tier="experimental",
    short_doc=(
        "Tempered Sequential Monte Carlo: a particle population annealed from "
        "the exact standardized N(0, I) prior to the posterior, so it never "
        "starts at the MAP and cannot inherit its basin. Every particle takes "
        "the same fixed-length inner-HMC moves at every rung, so a rung is "
        "lock-step with no ragged control flow -- but under the adaptive "
        "schedule the RUNG COUNT is data-dependent, so the raggedness moves out "
        "to the tempering while_loop rather than disappearing (pass "
        "fixed_ladder= for the fully lock-step arm). log Z comes free with the "
        "weights. NOTE: a resampled particle population is exchangeable, so the "
        "autocorrelation ESS is not a convergence count for this backend -- "
        "read min_ancestor_ess and the split R-hat across independent "
        "populations. Measured in bench/reports/2026-08-31_smc_evaluation.md."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_smc)

register_backend(
    "mcmc_barker",
    tier="experimental",
    short_doc=(
        "Barker-proposal MCMC: one gradient per step, no trajectory, no tree, "
        "and no branch anywhere in the compiled program -- two fixed-length "
        "lax.scan calls. Its published claim is robustness to a step size that "
        "is wrong for one direction's scale, which is this posterior's shape. "
        "Metric from precondition=, never estimated. Measured against its own "
        "MALA control in bench/reports/2026-08-31_blackjax_sampler_survey.md."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_barker)

register_backend(
    "mcmc_mala",
    tier="experimental",
    short_doc=(
        "MALA. Exists as the CONTROL for mcmc_barker -- same code path, same "
        "step-size-only dual averaging, same identity mass matrix, differing "
        "in the proposal alone -- so Barker's robustness claim is testable "
        "rather than asserted. Registered rather than hidden behind an edit "
        "because an ablation reachable only from source is one nobody re-runs."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_mala)

register_backend(
    "mcmc_hmc_lowrank",
    tier="experimental",
    short_doc=(
        "Fixed-L HMC whose mass matrix is a rank-k correction to a diagonal, "
        "fitted from warmup draws AND gradients by Fisher divergence. The "
        "middle term preconditioning.py's docstring says is missing between a "
        "diagonal that cannot cover cond 1e5 and a dense one that is noisy and "
        "memory-hungry. Sampling kernel and compile cost identical to mcmc_hmc, "
        "so a head-to-head isolates the mass matrix."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_hmc_low_rank)

register_backend(
    "mcmc_adjusted_mclmc",
    tier="experimental",
    short_doc=(
        "Adjusted microcanonical Langevin (cold ~60s, ~3x compile premium over "
        "mclmc). Requires blackjax >= 1.6."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_adjusted_mclmc)

register_backend(
    "mcmc_ess",
    tier="experimental",
    short_doc=(
        "Elliptical slice sampling, cheap (cold ~10s, ~2 GB) but assumes a "
        "Gaussian prior; bias on uniform/bounded priors not yet validated"
    ),
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_elliptical_slice)

register_backend(
    "vi_fullrank",
    tier="experimental",
    short_doc=(
        "Full-rank Gaussian VI (BlackJAX fullrank_vi): SGD on the ELBO over a "
        "Cholesky factor, so the fitted Gaussian CAN carry the age/dust/met "
        "tilt that mean-field cannot. A fixed-length scan, no ragged control "
        "flow, so the graph is small. It does not under-disperse -- at 2000 "
        "steps on D=8 DPL photometry it went UNSTABLE, returning "
        "sfh_dpl_log_total_mass as 10.37 +- 1.61 against a converged NUTS "
        "reference's 11.96 +- 0.036: the stellar mass wrong by 1.6 dex with an "
        "error bar 45x too wide, and no diagnostic in the family can report it. "
        "precondition=True is what makes it behave (worst width ratio 45x -> "
        "1.67x) and is effectively required. Check elbo_history and the width "
        "ratios in bench/reports/2026-08-31_vi_speed_evaluation.md before "
        "quoting anything from it."
    ),
    requires=("blackjax", "optax"),
    legacy_fitter=False,
    accepts_precondition=True,
)(_run_fullrank_vi)

register_backend(
    "vi_meanfield",
    tier="experimental",
    short_doc=(
        "Mean-field Gaussian VI (BlackJAX meanfield_vi): a DIAGONAL Gaussian, "
        "which on a posterior whose defining feature is a tilted degeneracy "
        "(cond 1e5-1e8) reports conditional widths rather than marginal ones. "
        "Measured against a converged NUTS reference on D=8 DPL photometry, its "
        "marginals are 0.07-0.62x the reference's, median 0.24x -- error bars 4x "
        "too narrow typically and 14x at worst. That is structural, not a tuning "
        "failure, and preconditioning does not fix it (median 0.21x). The "
        "cheapest member of the family and the least able to be right about an "
        "uncertainty; 'laplace' costs about the same and recovers widths to "
        "within 6%. See bench/reports/2026-08-31_vi_speed_evaluation.md."
    ),
    requires=("blackjax", "optax"),
    legacy_fitter=False,
    accepts_precondition=True,
)(_run_meanfield_vi)

register_backend(
    "nss",
    tier="experimental",
    short_doc=(
        "Nested sampling, slow (cold ~240s at D=6, timeout >600s at D=7); "
        "use for evidence/model comparison, not point estimates"
    ),
    legacy_fitter=False,
)(_ctx_run_nss)

register_backend(
    "pathfinder",
    tier="experimental",
    short_doc=(
        "[NARROW ERROR BARS] Pathfinder reports uncertainties up to 9x too small "
        "on a degenerate direction, silently, while looking healthy on every "
        "other parameter of the same fit: measured against a converged NUTS "
        "reference on D=8 DPL photometry, its marginal widths are 0.11-0.21x on "
        "the four degenerate SFH-shape parameters and 0.79-1.07x on the four "
        "well-constrained ones. Treat a Pathfinder error bar as a lower bound. "
        "The covariance is blackjax's lbfgs_inverse_hessian_formula_1, "
        "diag(alpha) + beta @ gamma @ beta.T: a diagonal plus a correction whose "
        "rank is however many L-BFGS steps actually ran -- but the cause is NOT "
        "the rank. Read out of a real fit, beta comes back FULL rank (8 of 8 at "
        "D=8) and 58% of the covariance's norm is off-diagonal, so the Gaussian "
        "is genuinely correlated. The cause is the ACCURACY of the quasi-Newton "
        "curvature estimate: its condition number is 8.7e3 against the analytic "
        "metric's 3.53e4 at the same point, 4x too isotropic. The clinching "
        "comparison is laplace, which fits the same family at the same expansion "
        "point with the EXACT Hessian and recovers widths at 0.94-1.05x on this "
        "fixture -- so neither the Gaussian family nor the MAP expansion point is "
        "at fault, only the estimate. Dropping the MAP seed, which blackjax's own "
        "page advises, makes it worse on 8 of 8 parameters (median 0.48 -> 0.16), "
        "so MAP seeding stays. "
        "precondition=True roughly halves the collapse (worst 0.11 -> 0.21) "
        "and does not cure it. There is also no R-hat and no divergence count to "
        "catch it, and the hierarchical seam OOM-kills it at D=18. What it is "
        "good for: a fast approximate CENTER (means within z ~ 1-11 of the same "
        "reference) and a NUTS warm start. Left tier='broken' 2026-05-22 to "
        "2026-08-31 under 'segfaults on DPL/dense_basis photometry mocks' (#231) "
        "-- a label the harness inferred from 'child died without writing JSON' "
        "without ever reading the child's return code; re-measured on that same "
        "model family it completes on every run, including at the uncapped 200 "
        "ELBO draws that were the pre-2026-07 default. Both real defects were "
        "fixed weeks later in PRs about other things (blackjax >= 1.4 API drift, "
        "4c1002ae7; uncapped ELBO draws at 26 GB, 8807c838d). See "
        "bench/reports/2026-08-31_vi_speed_evaluation.md."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
    accepts_precondition=True,
)(_ctx_run_pathfinder)
