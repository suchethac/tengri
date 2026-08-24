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
    run_dynamic_hmc as _ctx_run_dynamic_hmc,
    run_ghmc as _ctx_run_ghmc,
    run_hmc as _ctx_run_hmc,
    run_hmc_is as _ctx_run_hmc_is,
    run_mclmc as _ctx_run_mclmc,
    run_nuts as _ctx_run_nuts,
    run_raytrace as _ctx_run_raytrace,
)
from tengri.inference.backends.mcmc.elliptical_slice import (
    run_elliptical_slice_fitter as _ctx_run_elliptical_slice,
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
        "[POOR MIXING] Generalized HMC, fast (cold ~17s) but R-hat ≈ "
        "2.5-3.1 and ESS ≈ 1 on D=6-7 mocks even with 1000 warmup + 2000 "
        "samples. Do not use for science until adapter is fixed; see "
        "docs/dev/benchmarks/2026-05-22_inference_backend_validation.md."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_ghmc)

register_backend(
    "mcmc_mclmc",
    tier="broken",
    short_doc=(
        "[POOR MIXING] Microcanonical Langevin MC, fast warm call (~2s) "
        "but R-hat ≈ 1.7 / 1.13 and ESS ≈ 1 on D=6-7 mocks at 4000 samples. "
        "Do not use for science until tuning is investigated. "
        "Requires blackjax >= 1.6."
    ),
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_mclmc)

# ── Experimental backends ────────────────────────────────────────────────
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
    tier="broken",
    short_doc=(
        "[UNSTABLE] Pathfinder VI, segfaults on DPL/dense_basis photometry "
        "mocks (validated 2026-05-22, issue #231); use 'laplace' or 'vi' instead"
    ),
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_pathfinder)
