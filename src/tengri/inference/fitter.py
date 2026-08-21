# SPDX-License-Identifier: BSD-3-Clause
"""Inference engine: fit observed data using MAP, NUTS, Ray Tracing, or geoVI.

The Fitter separates inference strategy from the forward model. It builds
a loss function from the SEDModel's predictions and the Parameters's priors,
then runs the chosen optimizer/sampler.

Usage:
    from tengri import SEDModel, Fitter

    fitter = Fitter(model, data, noise)
    result_map = fitter.run("map", n_steps=1500)
    result_rts = fitter.run("raytrace", init_from=result_map)
    result_nuts = fitter.run("nuts", init_from=result_map, n_warmup=500)

Navigation
----------
This file stays one module by design; use the ``# ── <section> ──``
marker lines to jump. In order:

- ``Method name validation``: ``resolve_method`` (alias → canonical),
  legacy-SEDModel nudge, batched-data auto-extraction
- ``Fitter`` class (the bulk of the file):

  - ``Construction``: ``__init__`` (data validation, likelihood
    auto-build, emission lines, compile-cache setup)
  - ``Compilation``: ``compile_signature``, the JIT-engine cache,
    background compilation
  - ``Loss and likelihood builders``
  - ``Parameter transforms``
  - ``AOT pre-warm and adaptation persistence``
  - ``Inference dispatch``: ``run()`` routing into
    ``inference/backends/`` via the backend registry
  - ``Private method runners``
  - ``Posterior sampling``
  - ``Batch``

- ``Backend Registry Initialization``: imports that populate the
  registry at module load

"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tengri.inference._backend_registry import DEFAULT_METHOD
from tengri.inference._dimension_guard import warn_if_nuts_high_dim as _warn_if_nuts_high_dim

__all__ = ["Fitter", "resolve_method"]

if TYPE_CHECKING:
    from tengri.inference.posterior import Posterior

import jax
import numpy as np

logger = logging.getLogger(__name__)
import jax.numpy as jnp

from tengri.config.exceptions import ParameterError
from tengri.inference._model_cache import _default_owner as _model_cache_owner
from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical
from tengri.inference.jit_engine import build_jit_engine
from tengri.inference.likelihoods.gaussian import inv_noise_std
from tengri.inference.loss_functions import (
    build_loglikelihood_fn,
    build_loglikelihood_unbounded_fn,
    build_logprior_fn,
    build_loss_fn,
)
from tengri.parameters.priors import Gaussian, Uniform

# ── Method name validation ────────────────────────────────────────────

# D threshold for "auto": D <= this → mcmc_nuts, D > this → vi
_AUTO_D_THRESHOLD = 20

# D threshold for "mcmc": D <= this → NUTS, D > this → Ray Tracing
_MCMC_AUTO_D_THRESHOLD = 20

# Samplers that evaluate the forward model thousands of times per fit. On the
# exact wave-grid photometry path (no WavePrecomp LUT) each evaluation costs
# several times more (~2-6x measured, depending on which components are active),
# and the sampler pays that factor on the whole fit. Used to steer users to the
# fast path (see run()).
_MANY_EVAL_SAMPLERS = frozenset(
    {
        "mcmc",
        "mcmc_nuts",
        "mcmc_hmc",
        "mcmc_dynamic_hmc",
        "mcmc_ess",
        "mcmc_ghmc",
        "mcmc_mclmc",
        "mcmc_adjusted_mclmc",
        "nss",
        "native_vi_linear",
        "native_vi_nonlinear",
    }
)


def _prewarm_logger():
    """Logger for the best-effort prewarm paths.

    Prewarm legitimately catches everything — any exception a real fit can
    raise can surface during warmup, and narrowing the type would let some
    escape and abort a run that was going to succeed. So the catch stays broad
    and the failure becomes *visible* instead: a warmup that silently stopped
    working is indistinguishable from one that ran, and the only symptom is
    compile cost reappearing where it was supposed to have been paid already.

    ``debug`` rather than ``warning``: some configurations legitimately cannot
    prewarm, and a per-fit warning would be noise that teaches people to ignore
    it. Enable with ``logging.getLogger("tengri.inference.fitter").setLevel(
    logging.DEBUG)``.
    """
    import logging

    return logging.getLogger(__name__)


def _warn_if_exact_forward_path(model, backend_name: str) -> None:
    """Warn when a many-evaluation sampler runs on the exact photometry path.

    The WavePrecomp SSP x filter look-up table makes each forward evaluation
    several times cheaper, so an MCMC/nested sampler that calls the model
    thousands of times pays that factor on the whole fit.

    ``model`` is ``Fitter.model``, i.e. the model *after*
    :meth:`Fitter._resolve_fit_approx`. Under the default ``approx="auto"`` the
    fit is already routed through the LUT and this stays silent; it fires only
    when the exact path is genuinely in use (e.g. ``Fitter(..., approx=None)``).

    Ask ``model.approx`` — the public accessor both SEDModel and ForwardModel
    implement — never an inner attribute. ``Fitter.model`` is a ForwardModel,
    which does not carry the lowered ``_approx`` dict itself; reading it off the
    wrapper silently yields "exact" and would warn on every wrapped model. The
    accessor delegates to the inner SED, so the question has one spelling and
    one answer. (Supersedes the private ``_effective_approx`` probe.)
    """
    if backend_name not in _MANY_EVAL_SAMPLERS:
        return
    has_phot = getattr(getattr(model, "observation", None), "photometry", None) is not None
    if has_phot and not model.approx.wave_precomp:
        import warnings

        warnings.warn(
            f"Fitting with '{backend_name}' on the exact forward path: this "
            f"sampler evaluates the model thousands of times, and photometry on "
            f"the exact path costs several times more per evaluation than the "
            f"WavePrecomp look-up table (~2-6x, depending on which components "
            f"are active). Drop 'approx=None' to use the default 'auto' policy, "
            f"or pass approx=WavePrecomp(), for much faster sampling.",
            stacklevel=3,
        )


# Canonical method names (public API)
_CANONICAL_METHODS = {
    # --- Variational inference: 6 canonical names ---
    "vi_nonlinear_fast",  # NIFTy geoVI, no logging overhead — default
    "vi_linear_fast",  # NIFTy MGVI, no logging overhead
    "vi_nonlinear",  # NIFTy geoVI, standard (with logging)
    "vi_linear",  # NIFTy MGVI, standard (with logging)
    "native_vi_nonlinear",  # Pure-JAX geoVI (not yet implemented)
    "native_vi_linear",  # Pure-JAX MGVI (lax.while_loop, fastest)
    # "vi" kept as canonical synonym for vi_nonlinear (backward compat)
    "vi",
    "mcmc",  # auto: NUTS (D≤20) or Ray Tracing (D>20)
    "mcmc_raytrace",
    "mcmc_nuts",
    "mcmc_hmc",
    "mcmc_dynamic_hmc",
    "mcmc_ghmc",
    "mcmc_mclmc",
    "mcmc_adjusted_mclmc",
    "mcmc_ess",
    "map",
    "laplace",
    "pathfinder",
    "nss",  # Nested Slice Sampling, log Z (D≤30)
    "hmc_is",  # HMC posterior + importance-sampled log Z
    "auto",  # auto: mcmc_nuts (D≤20) or vi (D>20)
}


def _maybe_warn_legacy_sedmodel(model) -> None:
    """Nudge users from ``Fitter(sed_model, ...)`` to ``Fitter(forward, ...)``.

    Inference is canonically through :class:`ForwardModel` (issue #211).
    Passing a bare :class:`SEDModel` keeps working — it's the legacy
    pattern most existing notebooks use — but emits a one-shot
    :class:`DeprecationWarning` pointing at the canonical surface.

    :class:`ForwardModel` instances pass through silently (they ARE
    the canonical surface). Anything else (a likelihood Protocol, a
    test stub, …) also passes through silently — we don't want to
    warn on legitimate non-SEDModel uses.
    """
    try:
        from tengri.forward.forward_model import ForwardModel
        from tengri.forward.sed_model import SEDModel
    except ImportError:
        return
    if isinstance(model, ForwardModel):
        return
    if isinstance(model, SEDModel):
        import warnings

        warnings.warn(
            "Fitter(sed_model, ...) is deprecated and will be removed in "
            "tengri v1.0. Inference is canonically through ForwardModel "
            "(issue #211). Replace with: forward = ForwardModel.build("
            "sed=sed_model, observation=obs); Fitter(forward, data, noise)."
            "run(method) -- or use the shortcut forward.fit(data, noise, "
            "method=...).",
            DeprecationWarning,
            stacklevel=3,
        )


def _population_sed(model):
    """The :class:`PopulationSEDModel` inside ``model``, or ``None``.

    Parameters
    ----------
    model : object
        Candidate forward model.

    Returns
    -------
    PopulationSEDModel or None
        The population SubModel when ``model`` is a ``ForwardModel`` wrapping
        exactly one, otherwise ``None``.
    """
    try:
        from tengri.forward.forward_model import ForwardModel
        from tengri.forward.population_sed_model import PopulationSEDModel
    except ImportError:
        return None

    if not isinstance(model, ForwardModel):
        return None
    populations = getattr(model, "populations", ())
    if len(populations) != 1:
        return None
    pop_sed = getattr(populations[0], "sed", None)
    return pop_sed if isinstance(pop_sed, PopulationSEDModel) else None


def _maybe_extract_batched_data(model):
    """Auto-extract ``(data, noise)`` from a ForwardModel's population.

    Returns ``(None, None)`` if ``model`` is not a ForwardModel-with-
    PopulationSEDModel, otherwise the stacked ``(N, n_filters)`` arrays
    from ``pop.batched_data()``. Lets users write
    ``Fitter(forward).run('vi')`` without manually stacking.

    Explicit ``data=`` and ``noise=`` always override this default —
    auto-extraction only fires when both are ``None``.
    """
    pop_sed = _population_sed(model)
    if pop_sed is None:
        return None, None
    return pop_sed.batched_data()


def resolve_method(method: str, emit_warning: bool = True) -> str:
    """Validate that ``method`` is a canonical inference method name.

    Parameters
    ----------
    method : str
        Method name: canonical (e.g. ``"vi"``, ``"mcmc_nuts"``), ``"auto"``,
        or invalid.
    emit_warning : bool, optional
        Unused; retained for signature compatibility.

    Returns
    -------
    str
        The method name unchanged (canonical or ``"auto"``).

    Raises
    ------
    ParameterError
        If the method is not in :data:`_CANONICAL_METHODS` and not
        ``"auto"``. The error message lists every valid canonical name
        so the user can pick the intended one.
    """
    del emit_warning  # signature kept for backward source compatibility
    if method is None:
        raise ParameterError(
            "method=None is not allowed. Pass an explicit method string "
            "(e.g. 'vi', 'mcmc_nuts', 'auto') or omit the argument to use "
            "the default from defaults.toml."
        )

    if method in _CANONICAL_METHODS:
        return method

    canonical_list = ", ".join(sorted(_CANONICAL_METHODS))
    raise ParameterError(
        f"Unknown method: '{method}'. Valid names: {canonical_list}. "
        f"See Fitter.run() docstring for details."
    )


#: Constructor parameters the convenience fit surfaces manage themselves
#: (positionally or via their own named parameters) — never routed from a
#: surface's ``**kwargs``.
# Names the fit surface supplies from its own call signature. These can never be
# taken from ``**kwargs`` -- ``fit(data, noise, ...)`` already owns them.
_FIT_SURFACE_POSITIONAL = frozenset({"self", "model", "data", "noise"})

# Names the surface may DERIVE (e.g. ``data_type="joint"`` for a joint ``Data``
# record) but which are still ordinary ``Fitter.__init__`` parameters a caller may
# set explicitly. They must land in ``ctor_kwargs``.
#
# These used to sit in one undifferentiated set with the positional names, and
# ``split_fitter_kwargs`` excluded the whole set from ``ctor_names``. The effect
# was that passing any of them routed the value to ``Fitter.run()``, which hands
# it to the backend runner -- so a documented kwarg raised
# ``run_map() got an unexpected keyword argument 'data_type'`` (#1500). The
# derivation still wins by ``setdefault``, so an explicit value takes precedence
# without the surface losing its default.
_FIT_SURFACE_DERIVED = frozenset({"data_type", "data_mask", "approx", "params_override"})

_FIT_SURFACE_MANAGED = _FIT_SURFACE_POSITIONAL | _FIT_SURFACE_DERIVED


def _model_catalog_z_range(model):
    """The model's ``catalog_z_range`` (runtime-redshift LUT span), or ``None``.

    Reads it from a ``ForwardModel`` (via its population's SED) or a bare
    ``SEDModel`` — the same resolution the catalog engine uses.
    """
    if hasattr(model, "populations"):
        try:
            return model.populations[0].sed._catalog_z_range
        except (AttributeError, IndexError):
            return None
    return getattr(model, "_catalog_z_range", None)


def fit_surface_ctor_names() -> frozenset[str]:
    """Names the fit surfaces accept but route to ``Fitter.__init__``.

    Returns
    -------
    frozenset of str
        The constructor's keyword parameters, minus the ones the surfaces
        pass positionally.

    Notes
    -----
    Read off the live signature so it cannot drift from the constructor.
    :func:`split_fitter_kwargs` uses it to decide routing;
    ``tengri.inference._backend_registry.check_unknown_kwargs`` uses it
    to suggest a documented ``fit()`` option that is not a runner parameter.
    Both must mean the same thing by "constructor-routed", which is why this
    is one function rather than the same comprehension written twice.
    """
    import inspect

    return frozenset(
        name
        for name in inspect.signature(Fitter.__init__).parameters
        if name not in _FIT_SURFACE_POSITIONAL
    )


def split_fitter_kwargs(kwargs):
    """Split a fit-surface ``**kwargs`` dict into (constructor, run) halves (#1378).

    The convenience surfaces (``ForwardModel.fit``, ``SEDModel.fit``) accept one
    ``**kwargs``; parameters declared by ``Fitter.__init__`` — e.g.
    ``calibration_marginalize``, ``cal_n_poly``, ``eline_marginalize``,
    ``likelihood`` — belong to construction (spec #1320 §7 teaches them at the
    fit call), everything else to :meth:`Fitter.run`. The allowlist is derived
    from the live constructor signature so it cannot drift when the
    constructor gains parameters. Parameters the surfaces manage themselves
    (``data``, ``noise``, ``data_type``, ``data_mask``, ``approx``,
    ``params_override``) are never routed.

    Parameters
    ----------
    kwargs : dict
        The surface's collected ``**kwargs``. Not mutated.

    Returns
    -------
    ctor_kwargs : dict
        The subset belonging to ``Fitter.__init__``.
    run_kwargs : dict
        Everything else, for ``Fitter.run`` (unknown names still fail loudly
        there, as before).
    """
    ctor_names = fit_surface_ctor_names()
    ctor_kwargs = {k: v for k, v in kwargs.items() if k in ctor_names}
    run_kwargs = {k: v for k, v in kwargs.items() if k not in ctor_names}
    return ctor_kwargs, run_kwargs


# Fit-time approx clones, memoized per (source model, resolved config).
#
# Every ``Fitter`` resolves ``approx`` and clones the model, so N sequential
# per-galaxy fits over one ``ForwardModel`` produced N distinct clone objects.
# The compile caches (``_model_cache``, and the flat log-density built on it) key
# on model **identity**, so each galaxy missed the cache and recompiled — even
# though their ``_engine_cache_key()`` values were already identical. Returning
# the *same* clone for the same (source, config) makes those identity-keyed caches
# hit, without touching what any cache key means: same object implies same
# structure, so this cannot introduce the wrong-reuse hazard that re-keying
# structurally would (#1329 is what that looks like when it goes wrong).
#
# Safe to share because a resolved model is never mutated: nothing assigns to
# ``self.model.*`` or ``model.spec.*`` anywhere in this module, and per-fit state
# (``_params_override``, data, noise) lives on the Fitter.
#
# Keyed weakly on the source model, so the entry dies with the user's model and
# clones are not pinned. Mirrors ``_model_cache``'s WeakKeyDictionary, which
# already establishes that models are hashable and weak-referenceable.
_APPROX_CLONE_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _memoized_approx_clone(model, cfg):
    """``model.with_approx(cfg)``, returning one shared clone per distinct config.

    Parameters
    ----------
    model : SEDModel or ForwardModel
        Source model. Never mutated; used as the weak cache key.
    cfg : precompute config or tuple
        The **resolved** configuration. Keyed on this rather than on the caller's
        ``approx`` argument because resolution depends on fitter state (whether
        the fit has a line channel), so ``approx="auto"`` can legitimately resolve
        to different configs for different fits.

    Returns
    -------
    SEDModel or ForwardModel
        The clone for ``(model, cfg)`` — identical object across calls.

    Notes
    -----
    Not JIT-related itself; it exists so that downstream identity-keyed compile
    caches hit. A model that is unhashable or not weak-referenceable falls back to
    cloning every time, which is the previous behavior rather than an error.
    """
    try:
        bucket = _APPROX_CLONE_CACHE.setdefault(model, {})
    except TypeError:
        return model.with_approx(cfg)
    # repr, not hash: the precompute configs are dataclasses whose repr covers
    # every field, and not all of them are guaranteed hashable.
    key = (type(cfg).__name__, repr(cfg))
    clone = bucket.get(key)
    if clone is None:
        clone = model.with_approx(cfg)
        bucket[key] = clone
    return clone


def _component_chains(model) -> tuple:
    """Every component chain ``model`` owns, or ``()`` if none can be inspected.

    ``_build_component_chain`` lives on :class:`SEDModel` and nowhere else, so a
    bare ``getattr`` on the object the fitter is holding finds it only when that
    object *is* an ``SEDModel``. It usually is not: inference is canonically
    through :class:`ForwardModel` (#211), which holds ``populations[i].sed`` —
    the shape ``inference/catalog.py`` already reaches through by hand.

    That made :func:`fast_nebular_can_engage` answer ``True`` for every
    ``ForwardModel``, dusty or not — so the **photometry** gate it exists to
    enforce (#1748: a dusty model may not zero ``sed_nebular``, so the per-Q_H
    grid cannot serve photometry) never applied on the canonical path. Verified
    before and after on a dusty two_component model: the bare ``SEDModel``
    answered ``False``, the ``ForwardModel`` wrapping that same SED ``True``
    (#1790).

    This is the photometry question only. Whether a *line-flux* fit gets the LUT
    is a different question with a different answer — dust does not disarm that
    half, and #1770 measured 4.77x on a dusty line fit — so a caller deciding the
    line channel must not consult the predicate this feeds.

    Every population is consulted and :func:`fast_nebular_can_engage` requires
    all of them to be clear, rather than reading ``populations[0]`` — picking one
    arbitrarily is the failing-open shape ``ForwardModel._single_inner_sed``
    already refuses for the same reason (#1271).
    """
    chain = getattr(model, "_cached_component_chain", None)
    if chain is not None:
        return (chain,)

    builder = getattr(model, "_build_component_chain", None)
    if builder is not None:
        return (builder(),)

    populations = getattr(model, "populations", None) or ()
    chains: list = []
    for pop in populations:
        sed = getattr(pop, "sed", None)
        if sed is None:
            return ()
        inner = _component_chains(sed)
        if not inner:
            return ()
        chains.extend(inner)
    return tuple(chains)


def fast_nebular_can_engage(model) -> bool:
    """Can the fast nebular grid serve **photometry** for this model?

    Narrower than its original name suggests, and the difference cost a measured
    4.77x (#1770): this answers one of ``FeaturePrecomp``'s two jobs, not both.

    * **Photometry**: served from a per-Q_H grid, which requires zeroing
      ``sed_nebular``. Since #1281 that is only permitted when nothing downstream
      reads the continuum, and ``DustSEDComponent`` declares it as an input, so
      **any model with dust disarms this shortcut entirely**. That is what this
      predicate reports, and why #1748 stopped attaching the config for a
      photometry-only fit.
    * **A line channel**: served by supplying the line fluxes from the table, so
      ``loss_functions`` need not set ``needs_state=True`` and rebuild the
      full-grid SED through ``predict_state`` on every likelihood evaluation.
      **Dust does not touch this**, and this predicate says nothing about it.

    Consult it for the photometry top-up. Do **not** consult it to decide whether
    a line-flux fit gets the LUT: #1760 did, on the strength of a guard that
    measured ``jnp.sum(model.predict_photometry(params))`` — a photometry
    objective, which cannot observe the line-channel saving — and every dusty
    line-flux fit silently went back to the pre-#1477 cost. Measured on the
    #1477 fixture, gradient FLOPs of the fit objective: 1,933,823 with
    ``WavePrecomp`` alone against 405,825 with the LUT added, beside a dust-free
    control identical to the digit either way.

    Measured on main, one run, dust-free control in the same run, gradient FLOPs off
    the compiled HLO:

    ==========  ==============  ==============  ========
    model       ``WavePrecomp`` ``+Feature``    ratio
    ==========  ==============  ==============  ========
    with dust   65,438,628      65,438,628      **1.00x**
    no dust     54,827,036       1,789,868      30.63x
    ==========  ==============  ==============  ========

    Exact FLOP equality is the signature of a config that never reaches the graph,
    not of a lever with little left to pull (#1748).

    This is not a regression to undo. On the pre-#1281 tree the fast pair's photometry
    for a dusty model differed from exact by **0.41 %**, against 0.0115 % for a
    dust-free control — the shortcut was buying a biased answer, and a constant
    forward bias enters the gradient multiplied by SNR (#1671). What was wrong was
    advertising a speedup that no longer existed.

    Parameters
    ----------
    model : SEDModel
        The fit's model.

    Returns
    -------
    bool
        ``True`` when the grid could serve photometry. ``False`` for any model whose
        chain reads ``sed_nebular``.

    Notes
    -----
    Delegates to ``tengri.forward.sed_model._nebular_continuum_consumers``, the
    same expression ``enable_fast_nebular`` uses to set ``must_materialize_sed``, so
    the advice and the behavior cannot drift apart.
    """
    from tengri.forward.sed_model import _nebular_continuum_consumers

    chains = _component_chains(model)
    if not chains:
        # Deliberately permissive, and deliberately NOT changed with #1790.
        #
        # On the asymmetry alone this should fail closed: a wrong ``False`` only
        # forfeits a speedup, while a wrong ``True`` attaches a config #1748
        # measured as bit-identical in compiled FLOPs *and* changes
        # ``compile_signature()``, so the fit buys a second compiled kernel for
        # nothing. That was tried. Measured blast radius: 5 failures across
        # test_batch_inference_defaults_to_precomp, test_issue_1596_photometry_
        # feature_default and test_issue_1683_build_time_approx_feature_topup —
        # every one a ``_StubModel`` exposing neither a chain nor populations,
        # i.e. objects that are not models rather than models we cannot read.
        #
        # Once ``_component_chains`` unwraps the wrapper, every real surface
        # (SEDModel, ForwardModel, the batch and catalog paths) resolves to a
        # chain, so flipping this default has no demonstrated effect on any
        # production path — only on doubles. Rewriting five tests to enable a
        # hardening with no measured benefit is a separate change; #1790 records
        # it as a follow-up with that blast radius attached.
        return True
    return not any(_nebular_continuum_consumers(chain) for chain in chains)


def _observation_serves_line_channel(model) -> bool:
    """Whether the model's own Observation carries a measured line-flux channel.

    The model-visible half of :meth:`Fitter._fits_line_fluxes`, for the batch
    surfaces, which resolve their precompute policy without a ``Fitter`` and so
    cannot ask it. A fit that overrides the channel with ``line_flux_data=`` is a
    single-galaxy spelling and goes through the ``Fitter`` path, which resolves it
    properly; this is deliberately the *narrower* question.

    Parameters
    ----------
    model : SEDModel
        The fit's model.

    Returns
    -------
    bool
        ``True`` when ``model.observation`` declares ``line_fluxes``.
    """
    obs = getattr(model, "observation", None)
    return obs is not None and getattr(obs, "line_fluxes", None) is not None


def _has_line_adjacent_channel(model) -> bool:
    """Whether the observation carries a channel that reads line internals.

    Both members are now **measured**, not precautionary (#1665).

    ``line_ratios`` needs the nebular backend's DISCRETE line catalog
    (``line_waves``/``line_lums``). The precise mechanism is narrower than
    594a60552 recorded: ``FeaturePrecomp`` *alone* publishes that catalog
    fine — it is the ``WavePrecomp`` + ``FeaturePrecomp`` **pair** that drops
    it, because the per-Q_H photometry channel only exists when WavePrecomp
    supplied filters, and that is what arms the fast grid branch.

    ``spectral_indices`` was included as "unverified rather than known-broken".
    It is now known-broken and quantified: under the pair, all 13 indices move
    off the exact path — ``HgA`` by **+1733%**, ``Hbeta`` +734%, the Balmer
    family 29–1733% — because the fast grid zeroes ``nebular_sed`` and indices
    are measured on the rest-frame SED. So this gate is load-bearing; do not
    relax it for speed.

    ``line_fluxes`` is deliberately NOT here — that channel is the one
    ``FeaturePrecomp`` exists to serve. But a fit may carry line fluxes *and*
    an index/ratio channel, and then the LUT must still be refused: the
    ``_fits_lines`` top-up paths consult this predicate for exactly that case.

    Separate from :meth:`Fitter._fits_lines` because that predicate was being
    asked two *opposite* questions: :meth:`Fitter._auto_approx_config` appends
    the LUT *when* it is true, and the #1596 photometry top-up attaches the LUT
    when it is *false*. A photometry + line-ratio fit answered "no lines" to
    both, so it was classed photometry-only, handed the LUT, and then asked
    ``predict_line_ratios`` for a catalog the LUT does not publish. Widening
    ``_fits_lines`` would have fixed the second site by breaking the first —
    it would append the LUT for exactly the fits that cannot use it.
    """
    obs = getattr(model, "observation", None)
    if obs is None:
        return False
    return (
        getattr(obs, "line_ratios", None) is not None
        or getattr(obs, "spectral_indices", None) is not None
    )


#: Estimated relative posterior-gradient error above which the LUT bias warns.
#: 0.05 is #1671's own headline: the measured 0.13 % photometry bias crosses
#: it near SNR 30-40, i.e. exactly where the measurement showed the gradient
#: going materially wrong while every forward check stayed clean.
_LUT_BIAS_GRAD_WARN = 0.05


def _central_params(spec):
    """Free parameters at their declared prior medians, via ``unstandardize(0)``.

    The probe point for the LUT-bias estimate. ``unstandardize(0.0)`` is the
    declared prior's median for every distribution class — the same single
    source of truth the hierarchical seam standardizes through (#1651) — so
    no second defaults mechanism is invented. Stochastic-field latents probe
    at the zero draw (the field's own prior median).
    """
    params = {}
    for name in spec.free_params:
        params[name] = spec.get_distribution(name).unstandardize(0.0)
    if getattr(spec, "stochastic", False):
        params["sfh_field_xi"] = jnp.zeros(spec.n_grid)
    return params


def _lut_forward_bias(exact_model, lut_model, data_type):
    """Per-channel relative forward bias of the LUT, ``|lut - exact| / |exact|``.

    One exact and one LUT forward at the central parameters. Cached on the
    LUT clone keyed to the exact model's identity: a catalog constructs a
    fitter per galaxy against the same resolved clone, and the bias is a
    property of the model pair, not of the galaxy.

    Parameters
    ----------
    exact_model : SEDModel or ForwardModel
        The caller's un-resolved model — the exact path. Must be the SAME
        model the LUT clone was resolved from: pairing models that differ in
        anything but the LUT measures physics, not approximation.
    lut_model : SEDModel or ForwardModel
        The resolved clone.
    data_type : {"photometry", "spectroscopy"}

    Returns
    -------
    ndarray, shape (n_channels,)
        Relative bias per band / pixel [dimensionless].
    """
    cache = getattr(lut_model, "_lut_forward_bias_cache", None)
    if cache is not None and cache[0] is exact_model:
        return cache[1]
    params = _central_params(exact_model.spec)
    if data_type == "photometry":
        m_exact = np.asarray(exact_model.predict_photometry(params), dtype=float)
        m_lut = np.asarray(lut_model.predict_photometry(params), dtype=float)
    else:
        m_exact = np.asarray(exact_model.predict_spectrum(params), dtype=float)
        m_lut = np.asarray(lut_model.predict_spectrum(params), dtype=float)
    bias = np.abs(m_lut - m_exact) / np.maximum(np.abs(m_exact), np.finfo(float).tiny)
    # A frozen model just recomputes; the advisory still works.
    with contextlib.suppress(Exception):
        lut_model._lut_forward_bias_cache = (exact_model, bias)
    return bias


def _warn_if_lut_bias_amplified(exact_model, lut_model, data, noise, data_type, *, surface):
    """#1671's measurement made operational: warn when ``bias x SNR`` is material.

    The LUT's forward bias is constant in SNR, so no forward check can see
    it; the posterior gradient error it produces is ``~ bias x SNR`` (the
    chi-square gradient weighs the systematic offset by ``1/sigma``), moves
    the mode, and grows with data quality (#1671; the spectroscopy sibling
    was measured as a ~1-sigma posterior shift, #1688). This estimates
    ``max_i(bias_i x SNR_i)`` from one exact-vs-LUT forward on THIS model
    and this fit's data, and warns with the number and the remedy above
    :data:`_LUT_BIAS_GRAD_WARN`.

    Advisory contract: this function must never break a fit. Any failure in
    the probe (a forward that cannot run at the central parameters, shape
    mismatches, exotic data layouts) degrades to silence — the fit proceeds
    exactly as it did before the advisory existed. ``data_type="joint"`` is
    deliberately skipped: its data vector interleaves both channels and a
    wrong pairing would produce a wrong number, which is worse than none.

    Parameters
    ----------
    data, noise : array_like
        The fit's observed vector and 1-sigma noise, flattened; batch
        surfaces pass the per-galaxy concatenation.
    surface : str
        The fitting surface name, quoted in the warning.
    """
    if data_type not in ("photometry", "spectroscopy"):
        return
    try:
        bias = _lut_forward_bias(exact_model, lut_model, data_type)
        flat_data = np.asarray(data, dtype=float).reshape(-1)
        flat_noise = np.asarray(noise, dtype=float).reshape(-1)
        n = int(bias.shape[0])
        if n == 0 or flat_data.size == 0 or flat_data.size % n != 0:
            return
        snr = np.abs(flat_data) / np.maximum(flat_noise, np.finfo(float).tiny)
        est_all = bias[None, :] * snr.reshape(-1, n)
        i_flat = int(np.nanargmax(est_all))
        est = float(est_all.reshape(-1)[i_flat])
        channel = i_flat % n
        snr_at = float(snr[i_flat])
        bias_at = float(bias[channel])
    except Exception:
        return
    if not np.isfinite(est) or est <= _LUT_BIAS_GRAD_WARN:
        return
    from tengri.config.exceptions import PrecompBiasWarning, warn_measured

    warn_measured(
        f"{surface}: the precompute LUT's forward bias, amplified by this "
        f"fit's SNR, gives an estimated relative posterior-gradient error "
        f"of {est:.0%} (worst channel {channel}: forward bias "
        f"{bias_at:.2%} at SNR {snr_at:.0f}). The bias is constant in SNR "
        f"— invisible to any forward check — but enters the gradient "
        f"multiplied by SNR, moves the mode, and better data makes it "
        f"worse (#1671; spectroscopy sibling measured in #1688). For "
        f"final inference at this SNR, rerun with approx=None (the exact "
        f"path) or compare the two posteriors. Filter PrecompBiasWarning "
        f"if this trade is deliberate.",
        PrecompBiasWarning,
        stacklevel=3,
        gradient_error_estimate=est,
        worst_channel=channel,
        forward_bias=bias_at,
        snr=snr_at,
    )


def _resolve_batch_fit_approx(model, approx, data_type):
    """Route a batch-fit model through the fit-time precompute policy.

    The batch surfaces' mirror of :meth:`Fitter._resolve_fit_approx`.
    Single-galaxy fits have defaulted to the LUT under ``approx="auto"``
    since that policy landed, but ``PopulationFitter`` consumed its
    ``model_factory`` output raw and ``CatalogFitter`` held the model it was
    given — so exactly the fits that evaluate the forward model the most
    (thousands of evaluations, times the catalog size) silently ran the
    exact wave-grid path at a measured ~2-6x per-evaluation premium. A
    hierarchical or catalog fit taking minutes instead of well under one is
    this gap's symptom.

    Parameters
    ----------
    model : SEDModel or ForwardModel
        The batch fit's model (or one built by its factory). Never mutated;
        ``with_approx`` clones, memoized via :func:`_memoized_approx_clone`.
    approx : "auto" or None or precompute config
        ``"auto"`` selects the LUT for the data type (photometry ->
        ``WavePrecomp``, spectroscopy/joint -> ``SpectrumPrecomp``) and
        respects a model that already carries it; ``None`` returns the model
        untouched (the exact path); anything else is handed to
        ``with_approx`` verbatim.
    data_type : str
        The fit's data type; selects the LUT under ``"auto"``.

    Returns
    -------
    SEDModel or ForwardModel
        The LUT-routed clone, or the input model where the policy says (or
        the model allows) nothing else. A ``with_approx`` failure warns and
        stays exact — never break a fit that worked, only make its cost
        visible.
    """
    if approx is None:
        return model
    if getattr(model, "with_approx", None) is None:
        return model

    if isinstance(approx, str):
        if approx != "auto":
            raise ValueError(
                f"approx={approx!r} not understood; use 'auto' (default), None "
                "(exact), or a precompute config (WavePrecomp/SpectrumPrecomp, "
                "or a tuple)."
            )
        from tengri.forward.sed_model import FeaturePrecomp, SpectrumPrecomp, WavePrecomp

        state = getattr(model, "approx", None)
        if data_type == "photometry":
            has_wave = state is not None and state.wave_precomp
            has_feature = state is not None and getattr(state, "feature_precomp", False)
            cfg = WavePrecomp()
            # Attempt the feature top-up first: for a backend that can
            # tabulate its features (Cue's per-Q_H grid replaces the emulator
            # call itself) this is the dominant lever — measured 1.45x warm /
            # 1.68x cold on a 2-galaxy Cue population MAP fit (A/A floor
            # 1.17x) and ~7x per-gradient on the single-galaxy #1596 model,
            # where WavePrecomp alone does not clear the noise floor at all.
            # A backend with nothing to tabulate raises; that raise IS the
            # detection, so the fallback keeps the wave LUT rather than
            # regressing to the raw model.
            #
            # Only when NO line-adjacent channel exists: the ratio term reads
            # the backend's DISCRETE line catalog ('line_waves'/'line_lums'),
            # which the feature-LUT path does not publish — measured red on
            # main at 594a60552 (test_ratio_term_constrains_fit) because the
            # first spelling checked line_fluxes only, the channel matrix's
            # classic unwritten cell (#1460/#1480/#1599). spectral_indices is
            # excluded as unverified rather than known-broken.
            #
            # #1683: a model that ALREADY carries WavePrecomp returned here
            # untouched until this branch was restructured, so a batch fit built
            # with approx=WavePrecomp() never reached the top-up either — the
            # mirror of the single-galaxy gap, and the worse half: population
            # and catalog fits evaluate the forward model the most. When the
            # wave LUT is already configured, only the feature LUT is appended;
            # re-appending WavePrecomp would duplicate it.
            #
            # #1748: and only when the fast path can ENGAGE. Since #1281 a chain
            # that reads ``sed_nebular`` — anything with dust — disarms the grid's
            # photometry shortcut, so the top-up is bit-identical in compiled FLOPs
            # while still changing ``compile_signature()``. Batch surfaces pay that
            # per resolved clone, so skipping it here is the larger of the two wins.
            # The "dominant lever" numbers above were measured on the pre-gate tree
            # and hold only for a model with no ``sed_nebular`` consumer.
            #
            # ...unless a LINE channel is present, which is the other thing the LUT
            # serves and which dust does not disarm. #1775 drew that distinction on
            # the single-galaxy resolver and left this one gated, so the two
            # surfaces disagreed on exactly one cell of the channel matrix — a
            # dusty catalog fit that carries line fluxes kept refusing the LUT that
            # the same model got as a single-galaxy fit. ``data_type`` names the
            # primary data array here, not the channel set, so "photometry" does
            # not mean "no lines" (#1770).
            if (
                not has_feature
                and not _has_line_adjacent_channel(model)
                and (_observation_serves_line_channel(model) or fast_nebular_can_engage(model))
            ):
                existing = tuple(getattr(model, "approx_configs", ()))
                extra = (FeaturePrecomp(),) if has_wave else (cfg, FeaturePrecomp())
                with contextlib.suppress(Exception):
                    return _memoized_approx_clone(model, (*existing, *extra))
            if has_wave:
                return model
        elif data_type in ("spectroscopy", "joint"):
            if state is not None and getattr(state, "spectrum_precomp", False):
                return model
            cfg = SpectrumPrecomp()
        else:
            return model
        existing = tuple(getattr(model, "approx_configs", ()))
        resolved = (*existing, cfg) if existing else cfg
    else:
        resolved = approx

    try:
        return _memoized_approx_clone(model, resolved)
    except Exception as exc:  # broad on purpose — never break a working fit
        import warnings

        warnings.warn(
            f"Could not enable the precompute LUT for this batch fit "
            f"({exc}). The fit is correct, only slower — a measured ~2-6x "
            f"per forward evaluation on the exact path.",
            UserWarning,
            stacklevel=3,
        )
        return model


# Jitted predict wrappers, memoized per (model, method name).
#
# ``jax.jit`` caches on the callable's identity, and ``model.predict_photometry``
# constructs a fresh bound method on every attribute access — so
# ``jax.jit(model.predict_photometry)`` is a different function each time and
# recompiles. Holding the wrapper keeps the identity stable across fits.
_PREDICT_JIT_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _memoized_predict_jit(model, name: str):
    """``jax.jit(getattr(model, name))`` with a stable identity across calls.

    Parameters
    ----------
    model : SEDModel or ForwardModel
        Fit model; the weak cache key.
    name : str
        Attribute name of the accessor to wrap, e.g. ``"predict_photometry"``.

    Returns
    -------
    callable
        The jitted accessor — the same object on every call for a given
        ``(model, name)``, so JAX reuses its compiled executable.

    Notes
    -----
    A model that is unhashable or not weak-referenceable falls back to a fresh
    ``jax.jit`` each call, which is the previous behavior rather than an error.
    """
    try:
        bucket = _PREDICT_JIT_CACHE.setdefault(model, {})
    except TypeError:
        return jax.jit(getattr(model, name))
    fn = bucket.get(name)
    if fn is None:
        fn = jax.jit(getattr(model, name))
        bucket[name] = fn
    return fn


class Fitter:
    """Inference engine for differentiable SED fitting with flexible method dispatch.

    Separates inference strategy from the forward model by building a loss
    function from the SEDModel's predictions and the Parameters' priors, then
    running the chosen optimizer/sampler. Supports point estimation (MAP,
    Laplace), gradient-free and gradient-based sampling (ESS, NUTS, Ray
    Tracing, MCMC), variational inference (geoVI, MGVI), and nested sampling
    (NSS) via a unified ``run(method)`` interface.

    Parameters
    ----------
    model : SEDModel
        Configured forward model with ``spec`` (Parameters), ``observation``
        (Photometry/Spectroscopy/etc.), and predictor methods.
    data : array_like, shape (n_data,)
        Observed data (photometric fluxes or spectra). Units match the model's
        ``observation`` configuration. [erg/s/cm²/Hz] for photometry.
    noise : array_like, shape (n_data,)
        1-sigma measurement uncertainties. Same shape and units as ``data``.
    data_type : str or None
        Data type indicator: ``"photometry"``, ``"spectroscopy"``, or
        ``"joint"``. If ``None`` (default), inferred from
        ``model.observation``. Explicit values override inference.
    data_mask : array_like or None
        Optional per-datum censoring flags (CIGALE-style limits):
        ``0`` = detected (Gaussian term), ``1`` = upper limit
        (``ln Φ((limit − model)/σ)``), ``-1`` = lower limit
        (``ln Φ((model − limit)/σ)``). The corresponding ``data`` entry
        carries the limit value. Default ``None`` (all detected).
        Boolean arrays are rejected: ``True`` would silently be read as
        "upper limit", the opposite of the include/exclude semantics a
        boolean mask suggests. To exclude a datum, drop it from ``data``
        and ``noise`` (or inflate its ``noise``).
    calibration_marginalize : bool, optional
        If ``True``, analytically marginalize over spectroscopic calibration
        polynomial coefficients (Chebyshev order 1--``cal_n_poly``) when
        computing spectroscopic log-likelihood. Only applies when
        ``data_type`` ∈ {``"spectroscopy"``, ``"joint"``}. Follows Prospector
        (Johnson et al. 2021). Default ``False``.
    cal_n_poly : int, optional
        Number of Chebyshev polynomial coefficients for calibration
        marginalization (order 1 through ``cal_n_poly``). Default ``3``.
    cal_prior_sigma : float, optional
        Standard deviation of Gaussian prior on each calibration coefficient.
        Default ``1.0``.
    eline_marginalize : bool or None, optional
        Whether to analytically marginalize emission line amplitudes.
        ``None`` (default) auto-detects from the model's ``Spectroscopy``
        config (checks ``eline_mode == "marginalized"``).
    eline_prior_type : str or None, optional
        Prior type for emission line marginalization: ``"flat"`` (uniform) or
        ``"cloudy"`` (grid-interpolated from Cloudy models).
        ``None`` auto-detects from ``Spectroscopy.eline_prior_type``.
        Default ``None``.
    compile_modes : tuple[str, ...] or str or None, optional
        Control background JIT compilation during ``__init__``. Accepted values:

        - ``None`` (default) → no background compile; first ``run()`` compiles
          lazily.
        - ``"auto"`` → inspect ``spec.stochastic`` and ``data_type`` to select
          sensible defaults: stochastic → ``("linear_resample", "nonlinear_update")``
          (VI modes); non-stochastic photometry → ``("mcmc_nuts",)``; otherwise
          → ``("mcmc_nuts",)``.
        - explicit ``tuple[str, ...]`` (e.g., ``("mcmc_nuts",)``) → queue exactly
          those modes in the background thread.
        - explicit ``str`` (e.g., ``"mcmc_nuts"``) → wrap into a 1-tuple
          ``("mcmc_nuts",)``.

        Compile modes are passed to ``compile(modes=...)`` and determine which
        inference engines are pre-JIT-compiled before the first ``run()`` call.
        See ``compile()`` docstring for valid mode names.

    Returns
    -------
    Fitter
        Fitter instance with loss function compiled and ready for inference.

    Attributes
    ----------
    model : SEDModel
        Reference to the input forward model.
    data : ndarray, shape (n_data,)
        Input data as JAX array.
    noise : ndarray, shape (n_data,)
        Input noise as JAX array.
    data_type : str
        Resolved data type (``"photometry"``, ``"spectroscopy"``, ``"joint"``).
    spec : Parameters
        Reference to ``model.spec``.

    Notes
    -----
    **JIT-compatibility**: Methods in this class are not JIT-compatible because
    they perform Python-level branching on method names and manage resources
    (thread compilation, caching). The *returned* loss function and sampler
    engines are fully JIT-compiled and reusable across galaxies.

    **Background compilation**: Background compilation is now opt-in via the
    ``compile_modes`` parameter (default ``None`` = no background thread).
    The first ``run()`` call will compile lazily. Set ``compile_modes="auto"``
    or ``compile_modes=("mcmc_nuts",)`` to spawn a daemon thread and pre-compile
    specified inference modes before ``run()`` is called (typically <1s if warm,
    or the full compile time on cold XLA). Set ``TENGRI_NO_BACKGROUND_COMPILE=1``
    in the environment to disable even when ``compile_modes`` is set (test
    environments).

    **Engine caching**: Compiled engines are cached on the Model object so that
    multiple Fitters created with the same Model but different data reuse the
    same XLA programs. Cache key depends on data_type, dimensionality, free
    parameter names, and feature flags (emission lines, calibration).

    References
    ----------
    .. [1] B. D. Johnson et al., "Prospector: Stellar Population Inference from
       Spectra and SEDs," ApJS, 254, 22 (2021).
       arXiv:2012.01426. https://doi.org/10.3847/1538-4365/abef67

    Examples
    --------
    Fit a single galaxy with geoVI (default). The fitter takes a
    :class:`~tengri.ForwardModel`, so wrap the SED chain first:

    >>> from tengri import Fitter, ForwardModel, SEDModel  # doctest: +SKIP
    >>> sed = SEDModel.build(ssp_data=ssp, observation=obs, **config)  # doctest: +SKIP
    >>> forward = ForwardModel.build(sed=sed, observation=obs)  # doctest: +SKIP
    >>> data = jnp.array([1.2, 0.8, 0.5])  # photometric fluxes
    >>> noise = jnp.array([0.1, 0.08, 0.06])
    >>> fitter = Fitter(forward, data, noise)  # doctest: +SKIP
    >>> result = fitter.run("vi", n_samples=100)  # doctest: +SKIP
    >>> print(result.params)  # doctest: +SKIP

    ``forward.fit(data, noise, method="vi", n_samples=100)`` is the same fit in
    one line. Build the ``Fitter`` yourself when you want to keep it — warm
    starts, a reused compilation cache, or several backends over one model:

    >>> result_map = fitter.run("map", n_steps=1000)  # doctest: +SKIP
    >>> result_mcmc = fitter.run(  # doctest: +SKIP
    ...     "mcmc_nuts", init_from=result_map, n_warmup=500
    ... )

    See the docstring of :meth:`run` for all available methods and their options.
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(
        self,
        model,
        data=None,
        noise=None,
        data_type=None,
        data_mask=None,
        presence=None,
        line_flux_data=None,
        calibration_marginalize=False,
        cal_n_poly=3,
        cal_prior_sigma=1.0,
        eline_marginalize=None,
        eline_prior_type=None,
        likelihood=None,
        auto_protocol_likelihood=True,
        use_components=False,
        compile_modes=None,
        cache=None,
        approx="auto",
        params_override=None,
    ):
        # ── Auto-extract batched data for hierarchical ForwardModels ─
        # When ``model`` is a ForwardModel whose SubModel publishes
        # batched_axes (e.g. PopulationSEDModel publishes {'galaxy': 0}),
        # the per-galaxy data already lives on the population. Auto-
        # extract (N, n_filters) arrays so users can write
        # ``Fitter(forward).run('vi')`` without manually stacking.
        if data is None and noise is None:
            data, noise = _maybe_extract_batched_data(model)

        # ── Soft deprecation: prefer ForwardModel as the model arg ──
        # Inference is canonically through ForwardModel (issue #211).
        # Direct SEDModel as the model arg keeps working but nudges
        # callers to the new pattern.
        _maybe_warn_legacy_sedmodel(model)

        # ── Validate data/noise for the standard (non-hierarchical) path ──
        if data is None or noise is None:
            raise ValueError(
                "Fitter(model, data, noise) requires data and noise for "
                "non-hierarchical fits. For hierarchical fits, pass a "
                "ForwardModel built with population=PopulationSEDModel(...) "
                "and the per-galaxy data lives on the population."
            )

        # ── User-supplied Likelihood (Protocol path) ────────────────
        # When non-None, replaces the built-in χ² dispatch. The user
        # owns the entire data-term math and is responsible for
        # tracking their own observed arrays. Calibration / e-line
        # marginalization are NOT applied automatically — wrap them
        # into the user likelihood if needed.
        self._user_likelihood = likelihood
        self._auto_protocol_likelihood = auto_protocol_likelihood

        # ── Orchestrator opt-in (2026-05) ───────────────────────────
        # When True, route forward predictions through
        # :meth:`SEDModel.predict_state` (the SEDComponent
        # chain) instead of the legacy fused ``predict_photometry`` /
        # ``predict_spectrum`` kernels. Default ``False`` preserves
        # existing inference behavior bit-for-bit. Spectroscopy has no
        # orchestrator bridge yet, so combining ``use_components=True``
        # with non-photometric data_type is rejected at construction.
        self.use_components = bool(use_components)

        # ── Compile cache (ADR-deepen Step C, 2026-05) ──────────────
        # Optional per-Fitter CompileCache instance. When None, fall back
        # to the module-level singleton. Allows CatalogFitter to thread
        # a single cache through multiple per-galaxy Fitter instances,
        # preventing cross-galaxy evictions. Users can also isolate
        # Fitters with separate caches to guarantee no shared state.
        if cache is None:
            from tengri.inference.jit_engine import _get_singleton_cache

            cache = _get_singleton_cache()
        self.cache = cache

        # ── Data validation ─────────────────────────────────────────
        self.model = model
        self.data = jnp.asarray(data)
        self.noise = jnp.asarray(noise)
        if data_mask is not None:
            data_mask = jnp.asarray(data_mask)
            # A boolean mask here is a semantics trap: the censored
            # likelihood reads 1 as "upper limit", so True/False intended
            # as include/exclude would silently censor every included
            # datum. Require the explicit trinary convention.
            if data_mask.dtype == jnp.bool_:
                raise ValueError(
                    "data_mask must use censoring flags (0=detected, 1=upper "
                    "limit, -1=lower limit), not booleans. To exclude a datum, "
                    "drop it from data/noise or inflate its noise."
                )
        self.data_mask = data_mask
        if presence is not None:
            presence = jnp.asarray(presence, dtype=jnp.float32)
            if presence.shape != self.data.shape:
                raise ValueError(
                    f"presence shape {presence.shape} does not match data shape {self.data.shape}"
                )
        self.presence = presence
        # Per-galaxy emission-line values (#1599). The Observation carries the
        # line *schema* -- names, wavelengths, and whether each is a limit --
        # which is shared across a catalog; only the measured values differ per
        # galaxy. Both the data args and the compile key read this through
        # ``_resolved_line_fluxes`` so they cannot disagree.
        self._line_flux_override = line_flux_data
        self.data_type = self._resolve_data_type(data_type, model)
        self.spec = model.spec

        if self.use_components and self.data_type not in ("photometry", "spectroscopy", "joint"):
            raise NotImplementedError(
                "Fitter(use_components=True) currently supports "
                f"data_type in (photometry, spectroscopy, joint); got {self.data_type!r}."
            )

        # ── Calibration ────────────────────────────────────────────
        self._calibration_marginalize = calibration_marginalize
        self._cal_n_poly = cal_n_poly
        self._cal_prior_sigma = cal_prior_sigma
        self._has_spectroscopy = self.data_type in ("spectroscopy", "joint")

        # ── Emission lines ─────────────────────────────────────────
        # Resolved before the approx policy below: the "auto" precompute
        # config appends FeaturePrecomp when emission lines are fit.
        self._init_emission_lines(model, eline_marginalize, eline_prior_type)

        # ── Fit-time approximation policy (2026-07) ─────────────────
        # Fits default to the fast precompute LUT (approx="auto"); model
        # *prediction* stays exact. ``with_approx`` returns a clone, so the
        # user's original model is untouched and the returned posterior
        # references this fit model. ``approx=None`` forces the exact
        # wave-grid path; an explicit config (or tuple) overrides. Spec:
        # docs/internal/specs/2026-07-15-fit-precomp-default-design.md.
        self.model = self._resolve_fit_approx(model, approx)
        # For the #1671 bias advisory in run(): when resolution produced a LUT
        # clone, the caller's un-resolved object is the exact reference. When
        # the caller built the LUT into the model themselves there is no exact
        # reference and the check is skipped (with_approx(None) is not a safe
        # substitute — see #1671's measurement-hygiene note). Deferred to
        # run() so that merely constructing a fitter stays cheap: the probe
        # costs one exact forward, and only an executed fit should pay it.
        self._pre_approx_model = model if self.model is not model else None
        self._lut_bias_checked = False
        self.spec = self.model.spec

        # Re-apply the fitted-mode emission-line amplitude merge. The
        # reassignment above replaces self.spec with the approx-resolved
        # model's spec, which does not carry the ``eline_amp_*`` priors that
        # _init_emission_lines merged into the pre-approx spec. Without this,
        # the amplitudes silently drop out of ``free_params`` (and thus
        # ``_free_names``) while ``_eline_amplitude_names`` still lists them —
        # the sampler never varies parameters the fitter believes it is
        # fitting. See tests/inference/test_eline_fitting.py::TestFittedMode.
        if getattr(self, "_eline_amp_priors", None):
            self.spec = self.spec.merge_observation_params(**self._eline_amp_priors)

        # ── Parameters ─────────────────────────────────────────────
        self._free_names = self.spec.free_params
        self._fixed_values = self.spec.get_fixed_values()
        self._bounds = {n: self.spec.get_distribution(n).bounds for n in self._free_names}

        # ── Per-fit params override (issue #1329) ──────────────────
        # Validate params_override: keys must be fixed parameters (not free),
        # and must be valid parameter names.
        self._params_override = None
        if params_override is not None:
            self._params_override = dict(params_override)
            for key in self._params_override:
                if key in self._free_names:
                    raise ValueError(
                        f"Parameter {key!r} is free and cannot be overridden in a fit. "
                        f"A params= key that names a free parameter (being fit) would "
                        f"corrupt inference. Free parameters: {sorted(self._free_names)}"
                    )
                if key not in self._fixed_values and key not in self._free_names:
                    all_params = sorted(set(self._free_names) | set(self._fixed_values.keys()))
                    raise ValueError(
                        f"Parameter {key!r} is not a valid parameter name. "
                        f"Valid parameters: {all_params}"
                    )
            # Merge the override INTO the fixed-values dict — this is the single
            # source of truth the loss closure bakes at build time
            # (``loss_functions.build_loss_fn`` -> ``fitter._fixed_values``) and the
            # output conversion (``_to_physical``) reads. Merging only in
            # ``_to_physical`` is a silent relabel: the loss keeps running at the
            # model's fixed value while the returned params echo the override.
            # The override is part of ``_engine_cache_key`` so a second fit with a
            # different override compiles its own loss rather than reusing this one.
            self._fixed_values = {**self._fixed_values, **self._params_override}

        # ── Runtime redshift routing (#1316, spec §9.4) ────────────
        # On a model whose ztable spans a ``catalog_z_range``, a redshift
        # override is a RUNTIME input to the LUT interpolation — thread it
        # through ``data_args`` (the #1349 seam: ``build_loss_fn`` replaces the
        # baked value with ``data_args["redshift"]``) and keep it OUT of the
        # engine cache key, so distinct per-row redshifts share one compiled
        # program instead of recompiling per row. The merge into
        # ``_fixed_values`` above still happens — it keeps reporting
        # (``_to_physical``) honest, and the baked value is dead weight in the
        # loss because the ``data_args`` injection always overrides it.
        # Invariant: the key omits redshift *iff* ``data_args`` carries it, so
        # a shared loss closure can never silently run at another fit's baked z.
        # Overrides on models without a ztable keep #1331's bake — there the
        # redshift genuinely is a compile constant.
        self._runtime_redshift = None
        if (
            self._params_override is not None
            and "redshift" in self._params_override
            and _model_catalog_z_range(model) is not None
        ):
            self._runtime_redshift = float(self._params_override["redshift"])

        # ── Data arguments ─────────────────────────────────────────
        self._data_args = self._build_data_args(model)

        # ── Auto-build Protocol likelihood (option β default) ──────
        # When the user didn't pass a custom likelihood AND none of the
        # legacy-only features (cal-marg, e-line marg, Student-t,
        # spec-cov, censored, line fluxes, indices) are configured,
        # build the matching :class:`Likelihood` Protocol object
        # (Photometry / Spectroscopy / Composite) from data + noise.
        # This routes simple cases through the new path so the
        # diagonal-Gaussian χ² lives in exactly one place. Legacy
        # dispatch still handles any case that asks for an extra.
        if self._user_likelihood is None and self._auto_protocol_likelihood:
            self._user_likelihood = self._maybe_build_default_likelihood()

        # ── Memory-mode auto-detect ─────────────────────────────────
        # Pre-set _memory_mode before spawning the background compile
        # thread so that thread builds the correct engine variant the
        # first time. Without this, the thread reads the default "fast"
        # and compiles the wrong engine; run() then flips the mode and
        # triggers a second compile, holding both engines in the
        # model-level cache simultaneously.
        # The user can still override at run() time via memory_mode=...
        # (doing so invalidates _jit_sampler and triggers a rebuild).
        self._memory_mode = "low" if self.spec.stochastic else "fast"
        self._posterior_chunk_size = None

        # ── Background compilation modes ───────────────────────────
        # Process compile_modes parameter: None → (), "auto" → infer,
        # str → wrap, tuple → use as-is. Empty tuple skips background compile.
        self._target_modes = self._resolve_compile_modes(compile_modes)

        # ── Background compilation ─────────────────────────────────
        self._jit_sampler = None
        self._compilation_event = threading.Event()
        self._compilation_error: Exception | None = None
        self._compilation_lock = threading.Lock()
        self._compilation_thread: threading.Thread | None = None
        self._start_background_compilation()

    def __repr__(self) -> str:
        """One-line summary of how this fitter is configured."""
        n_free = len(self._free_names)
        n_fixed = len(self._fixed_values)
        n_data = int(self.data.shape[0]) if hasattr(self.data, "shape") else "?"
        dt = getattr(self, "data_type", "?")
        sfh = "stochastic" if self.spec.stochastic else "parametric"
        return (
            f"Fitter(data_type={dt!r}, n_data={n_data}, "
            f"n_free={n_free}, n_fixed={n_fixed}, sfh={sfh!r})"
        )

    def _maybe_build_default_likelihood(self):
        """Build the default Protocol likelihood for this Fitter's data.

        Now handles every case in the likelihood-Protocol cohort:

        - simple diagonal Gaussian → ``PhotometryLikelihood`` /
          ``SpectroscopyLikelihood``
        - joint phot+spec → ``CompositeLikelihood``
        - Student-t (variable noise) → ``StudentTLikelihood``
        - censored data → ``CensoredLikelihood``
        - spec covariance → ``MultivariateGaussianLikelihood``
        - calibration marginalization → ``CalibrationMarginalizedLikelihood``
        - flat-prior e-line marginalization → ``ELineMarginalizedLikelihood``
          (with a per-call design-matrix builder closure)
        - line fluxes / spectral indices → composed onto the base
          via ``CompositeLikelihood``

        Returns ``None`` only for cases the Protocol path does not yet
        cover — currently the Cloudy-prior e-line marginalization
        (uses a different math primitive) and the e-line *fitted*
        amplitudes (line amplitudes are fit, not marginalized).
        """
        from tengri.inference.composite_likelihood import CompositeLikelihood
        from tengri.inference.context import InferenceContext
        from tengri.inference.likelihood import (
            build_base_likelihood,
            build_likelihood_extras,
        )

        context = InferenceContext.from_target(self)
        base = build_base_likelihood(context)
        if base is None:
            return None
        extras = build_likelihood_extras(context)
        if not extras:
            return base
        return CompositeLikelihood(base, *extras)

    @staticmethod
    def _resolve_data_type(data_type: str | None, model: Any) -> str:
        """Infer data_type from Observation when not explicitly provided."""
        if data_type is not None:
            return data_type
        obs = getattr(model, "observation", None)
        if obs is not None:
            return obs.data_type
        return "photometry"

    def _line_fluxes_for(self, model):
        """The line-flux config this fit scores against, for ``model``'s schema.

        Parameters
        ----------
        model : SEDModel
            The model whose ``observation`` carries the line schema.

        Returns
        -------
        LineFluxData or None
            The ``line_flux_data=`` override when one was supplied, else the
            Observation's own ``line_fluxes``.

        Notes
        -----
        THE one place a line-flux channel is resolved, because a channel with
        two sources read separately is a channel two callers can disagree
        about. The approx predicate read the attribute alone, so a fit
        supplying its fluxes per galaxy (``line_flux_data=``, #1599) published
        ``line_flux_waves`` and set ``has_line_fluxes`` in the loss while
        classifying as photometry-only — and on a model built with
        ``approx=WavePrecomp()`` it was then handed back untouched, losing the
        ``FeaturePrecomp`` top-up at ~21x the per-gradient cost, with no
        warning. A third source means editing this function and nothing else.
        """
        if self._line_flux_override is not None:
            return self._line_flux_override
        obs = getattr(model, "observation", None)
        return getattr(obs, "line_fluxes", None) if obs is not None else None

    def _fits_line_fluxes(self, model) -> bool:
        """Whether this fit has a measured emission-line-flux channel.

        Resolves through :meth:`_line_fluxes_for` — the same call
        :meth:`_build_data_args` makes to publish ``line_flux_waves``, and hence
        what makes ``build_loss_fn`` set ``has_line_fluxes``. Single-sourcing
        the condition matters: the whole point is that the LUT is added when
        (and only when) the loss would otherwise pay for the full-grid forward,
        so the two must not be able to disagree.

        ``_data_args`` itself is not available here — it is built after the
        approx policy resolves — so the channel is resolved directly.
        """
        return self._line_fluxes_for(model) is not None

    def _resolved_line_fluxes(self):
        """The line-flux config this fit actually scores against.

        Returns
        -------
        LineFluxData or None
            The ``line_flux_data=`` override when one was supplied, else the
            Observation's own ``line_fluxes``.

        Notes
        -----
        Read by :meth:`_build_data_args` *and* by :meth:`compile_signature`.
        They must resolve identically: the signature keys on whether a limit
        mask is present, which selects the censored-vs-Gaussian adapter at
        build time, while the mask values ride through the data args. If the
        two read different objects, a fit can compile the Gaussian adapter and
        then be handed a censored mask (#1599).
        """
        return self._line_fluxes_for(self.model)

    def _fits_lines(self, model) -> bool:
        """Whether any emission-line channel is fit, measured or marginalized.

        Answers only "is there a line channel". Whether the LUT may *serve* it
        is :func:`_has_line_adjacent_channel` — see there for why the two are
        deliberately separate predicates.
        """
        return bool(
            getattr(self, "_eline_marginalize", False)
            or getattr(self, "_eline_fitted", False)
            or self._fits_line_fluxes(model)
        )

    def _auto_approx_config(self, model):
        """Precompute config auto-selected for this fit's data type.

        Photometry -> ``WavePrecomp``; spectroscopy/joint -> ``SpectrumPrecomp``;
        ``FeaturePrecomp`` is appended when emission lines are fit. Returns
        ``None`` for data types with no LUT mapping (the fit then stays exact).

        "Emission lines are fit" means **any** line channel: the spectroscopy
        nuisance amplitudes (``_eline_marginalize`` / ``_eline_fitted``) *and*
        a measured line-flux channel on the observation. Only the former were
        checked until 2026-07, so the channel most users mean — fitting
        ``LineFluxData`` alongside photometry — silently stayed on the exact
        path at ~21x the per-gradient cost.
        """
        from tengri.forward.sed_model import (
            FeaturePrecomp,
            SpectrumPrecomp,
            WavePrecomp,
        )

        if self.data_type in ("spectroscopy", "joint"):
            base = SpectrumPrecomp()
        elif self.data_type == "photometry":
            base = WavePrecomp()
        else:
            return None
        # A line-flux channel earns the LUT, but an index/ratio channel on the
        # SAME fit vetoes it: those read the rest-frame SED / discrete catalog,
        # which the fast grid does not serve (#1665). Such a fit pays the
        # full-grid forward regardless, so the LUT would be a wrapper around
        # the cost it exists to avoid, and a broken one. Correctness outranks
        # the line speedup, and the refusal is loud rather than silent.
        #
        # No ``fast_nebular_can_engage`` gate here, deliberately (#1770). That
        # predicate answers whether the grid may serve PHOTOMETRY without
        # materializing ``sed_nebular``, which dust disarms (#1748/#1281). This
        # branch serves a LINE channel, where the LUT's value is that the line
        # fluxes come from the table instead of ``needs_state=True`` forcing a
        # full-grid ``predict_state`` per likelihood — dust does not touch that.
        # Gating it here cost a measured 4.77x on every dusty line-flux fit.
        wants_lut = self._fits_lines(model) and not _has_line_adjacent_channel(model)
        return (base, FeaturePrecomp()) if wants_lut else base

    def _add_feature_precomp(
        self, model, *, warn_on_failure: bool = True, serves_line_channel: bool = True
    ):
        """Top up a build-time ``approx=`` with ``FeaturePrecomp``.

        A model built with ``approx=WavePrecomp()`` used to be returned
        untouched by the ``"auto"`` policy, so naming WavePrecomp explicitly
        made a fit *slower than passing nothing at all* — the exact opposite of
        what the argument reads like it does. That was fixed for a lines fit
        first and for a photometry-only one in #1683; both spellings now reach
        here.

        The existing configs are carried over rather than rebuilt, so a
        configured ``catalog_z_range`` survives; see
        :attr:`SEDModel.approx_configs`. Only ``FeaturePrecomp`` is appended —
        re-appending the wave LUT would duplicate it.

        Parameters
        ----------
        model : SEDModel
            The fit's model. Never mutated; ``with_approx`` clones.
        warn_on_failure : bool, keyword-only, optional
            Whether an unavailable LUT is worth telling the caller about.
            ``True`` for a line channel, where the documented cost of going
            without is ~21x per gradient. ``False`` for the photometry-only
            top-up, where a backend with nothing to tabulate is the *expected*
            case rather than an anomaly — warning there would fire on every
            non-Cue model and name a remedy the caller cannot apply.

        Returns
        -------
        SEDModel
            The topped-up clone, or ``model`` unchanged when the LUT is already
            configured or could not be enabled.

        Notes
        -----
        Adding the LUT can legitimately fail — a nebular backend that publishes
        neither a discrete catalog nor SSP-window lines has no fast path. That
        is a reason to stay exact, never to break a fit that worked, so the
        failure is caught either way; ``warn_on_failure`` decides only whether
        it is announced.
        """
        from tengri.forward.sed_model import FeaturePrecomp

        state = getattr(model, "approx", None)
        if state is None or state.feature_precomp:
            return model
        # One seam for every caller of this top-up (#1665): a model whose
        # observation also carries an index or ratio channel must stay off the
        # feature LUT, whatever route asked for it.
        if _has_line_adjacent_channel(model):
            return model
        # Nor top up a model the LUT cannot help — but "cannot help" is channel
        # specific, and conflating the two channels is what #1770 was (see
        # ``fast_nebular_can_engage``). The dust gate applies to the PHOTOMETRY
        # shortcut only: since #1281 a chain that reads ``sed_nebular`` disarms it,
        # so appending the config there buys a second compiled kernel for no effect
        # (#1748). A LINE channel is served by a different mechanism — the LUT
        # supplies the line fluxes directly, instead of ``needs_state=True`` driving
        # a full-grid ``predict_state`` rebuild per likelihood — and that mechanism
        # is unaffected by dust. Measured on the #1477 fixture, a dusty model with
        # three line fluxes, gradient FLOPs of the fit objective:
        # WavePrecomp 1,933,823 -> WavePrecomp+FeaturePrecomp 405,825, a 4.77x
        # reduction, against a dust-free control that is identical to the digit
        # (251,783 either way) because the top-up has already happened there.
        if not serves_line_channel and not fast_nebular_can_engage(model):
            return model
        existing = tuple(getattr(model, "approx_configs", ()))
        try:
            return _memoized_approx_clone(model, (*existing, FeaturePrecomp()))
        except Exception as exc:  # broad on purpose — never break a working fit
            if not warn_on_failure:
                return model
            import warnings

            warnings.warn(
                f"Fitting an emission-line channel, but the line look-up table "
                f"could not be enabled for this model ({exc}). Every likelihood "
                f"evaluation will reconstruct the full-wavelength SED to obtain "
                f"the line fluxes — measured at ~21x the per-gradient cost. The "
                f"fit is correct, only slow.",
                UserWarning,
                stacklevel=4,
            )
            return model

    def _resolve_fit_approx(self, model: Any, approx):
        """Select the fit-time forward model per the ``approx`` policy.

        - ``"auto"`` (default): route the fit through the precompute LUT chosen
          by data type (see :meth:`_auto_approx_config`). A build-time
          ``approx=`` is respected, but is **topped up** with ``FeaturePrecomp``
          when a line channel is fit and it is missing — otherwise naming
          ``WavePrecomp()`` at build time would make a lines fit slower than
          passing nothing at all.
        - ``None``: force the exact wave-grid path (overrides a build-time approx).
        - an explicit config / tuple: use exactly that.

        Only ``"auto"`` auto-activates. ``None`` means exact and stays exact; an
        explicit config means what it says. Both instead warn when a line
        channel is fit without the LUT, so the cost is visible rather than
        silent — the prior decision was against *silent* auto-activation, and a
        warning is how that is honored without leaving the cliff unmarked.

        ``model.with_approx`` returns a clone (or ``self`` for a no-op), so the
        user's original model object is never mutated. Models that cannot clone
        (no ``with_approx``) are returned unchanged.
        """
        if isinstance(approx, str) and approx != "auto":
            raise ValueError(
                f"approx={approx!r} not understood; use 'auto' (default), None "
                "(exact), or a precompute config (WavePrecomp/SpectrumPrecomp/"
                "FeaturePrecomp, or a tuple)."
            )
        with_approx = getattr(model, "with_approx", None)
        if not callable(with_approx):
            return model
        if isinstance(approx, str):  # "auto"
            has = getattr(model, "_has_modern_approx", None)
            if callable(has) and has():
                # Respect the build-time approx, but do not let it suppress the
                # line LUT — top it up rather than bailing.
                #
                # The lines branch deliberately does NOT repeat the ratio/index
                # exclusion that this branch carried at the call site before the
                # thirteenth merge. It lives inside ``_add_feature_precomp`` as
                # "one seam for every caller of this top-up" (#1665), so a
                # line-flux + ratio fit is still refused — just refused in one
                # place. Restating it here is what let the two spellings drift
                # apart in the first place.
                if self._fits_lines(model):
                    return self._add_feature_precomp(model, serves_line_channel=True)
                # #1683: #1596 in its BUILD-TIME spelling. The fix that landed
                # in #1656 tops up only the branch below, where the model carries
                # no approx at all; a photometry-only Cue model built WITH
                # approx=WavePrecomp() still returned here untouched. So
                # naming the wave LUT explicitly cost the feature LUT — the
                # same "slower than passing nothing at all" pathology this
                # method's docstring records for lines, measured at ~22x the
                # per-gradient FLOPs on a 10-parameter Cue model. Silent on
                # failure: a backend with nothing to tabulate is the common
                # case here, not an anomaly worth warning about.
                if self.data_type == "photometry" and not _has_line_adjacent_channel(model):
                    return self._add_feature_precomp(
                        model, warn_on_failure=False, serves_line_channel=False
                    )
                return model
            cfg = self._auto_approx_config(model)
            if cfg is None:
                return model
            if (
                self.data_type == "photometry"
                and not self._fits_lines(model)
                and not _has_line_adjacent_channel(model)
                # #1748: and only when the grid can actually serve photometry. A
                # chain that reads ``sed_nebular`` — anything with dust — disarms
                # the shortcut since #1281, making this append bit-identical in
                # compiled FLOPs while still changing ``compile_signature()``.
                # This branch appends the config directly rather than through
                # ``_add_feature_precomp``, so it needs the predicate of its own;
                # guarding only the helper left this path attaching it anyway.
                and fast_nebular_can_engage(model)
            ):
                # #1596: a photometry-only Cue fit measured ~4x SLOWER than
                # the same fit WITH a line channel, because FeaturePrecomp —
                # despite the name, the nebular precompute; for Cue the
                # per-Q_H grid replaces the emulator call itself (~7x
                # per-gradient) — was only added when lines were fit, and
                # WavePrecomp alone does not clear the noise floor on a Cue
                # model. Attempt the feature top-up; a backend with nothing
                # to tabulate raises, and that raise IS the detection — fall
                # back to the wave LUT, never to the raw model. Mirrors the
                # batch surfaces' _resolve_batch_fit_approx.
                from tengri.forward.sed_model import FeaturePrecomp

                base = cfg if isinstance(cfg, tuple) else (cfg,)
                with contextlib.suppress(Exception):
                    return _memoized_approx_clone(model, (*base, FeaturePrecomp()))
            return _memoized_approx_clone(model, cfg)

        resolved = _memoized_approx_clone(model, approx)
        self._warn_lines_without_lut(resolved)
        return resolved

    def _warn_lines_without_lut(self, model) -> None:
        """Warn when a line channel is fit on the exact path by explicit request.

        Fires for ``approx=None`` and for an explicit config that omits
        ``FeaturePrecomp`` — the two cases the ``"auto"`` policy deliberately
        does not override. Silence here is what let a 21x per-gradient cost look
        like the model simply being slow.
        """
        if not self._fits_lines(model):
            return
        if _has_line_adjacent_channel(model):
            # The LUT is withheld here on purpose — a ratio/index channel needs
            # the discrete catalog it does not publish. Warning would point at a
            # remedy that cannot be applied, and advice you cannot act on reads
            # as a defect in the caller's model.
            return
        # No ``fast_nebular_can_engage`` gate (#1770). It was added here on the
        # reading that a dusty model gains nothing from the LUT — true of the
        # photometry shortcut, false of this channel, where the saving is the
        # ``predict_state`` rebuild rather than the nebular grid. On a dusty
        # line-flux fit the advice IS actionable: 4.77x in gradient FLOPs.
        state = getattr(model, "approx", None)
        if state is not None and state.feature_precomp:
            return
        import warnings

        warnings.warn(
            "Fitting an emission-line channel without FeaturePrecomp: every "
            "likelihood evaluation reconstructs the full-wavelength SED just to "
            "obtain the line fluxes, measured at ~21x the per-gradient cost "
            "(6.95 ms vs 0.31 ms on a 5-band, 3-line model, with no dust). Pass "
            "approx=(WavePrecomp(), FeaturePrecomp()), or drop approx= to use "
            "the default 'auto' policy, which adds it for you.",
            UserWarning,
            stacklevel=4,
        )

    def _init_emission_lines(self, model, eline_marginalize, eline_prior_type):
        """Configure emission line marginalization and fitted-amplitude modes."""
        _spec_config = getattr(model, "_spectroscopy_config", None)
        if _spec_config is None:
            obs = getattr(model, "observation", None)
            if obs is not None:
                _spec_config = getattr(obs, "spectroscopy", None)

        # Marginalization mode
        if eline_marginalize is None:
            if _spec_config is not None and hasattr(_spec_config, "eline_mode"):
                eline_marginalize = _spec_config.eline_mode == "marginalized"
            else:
                eline_marginalize = False
        self._eline_marginalize = bool(eline_marginalize) and self._has_spectroscopy

        # Fitted emission line mode — amplitudes become explicit latent params
        if _spec_config is not None and hasattr(_spec_config, "eline_mode"):
            _eline_fitted = _spec_config.eline_mode == "fitted"
        else:
            _eline_fitted = False
        self._eline_fitted = bool(_eline_fitted) and self._has_spectroscopy

        # Prior type
        if eline_prior_type is None:
            if _spec_config is not None and hasattr(_spec_config, "eline_prior_type"):
                _raw = _spec_config.eline_prior_type
                eline_prior_type = _raw if isinstance(_raw, str) else "flat"
            else:
                eline_prior_type = "flat"
        self._eline_prior_type = eline_prior_type

        # Precompute static arrays for emission line fitting
        if self._eline_marginalize or self._eline_fitted:
            self._init_eline_arrays(_spec_config)
        else:
            self._eline_wavelengths = None
            self._eline_independent_wavelengths = None
            self._eline_names = None
            self._eline_constraint_matrix = None
            self._eline_prior_sigma = 100.0
            self._eline_prior_width_dex = 0.3
            self._eline_amplitude_names = []

        # Consistency check: Spectroscopy.eline_broad vs Parameters.eline_broad
        if _spec_config is not None and getattr(_spec_config, "eline_broad", False):
            spec_has_broad = getattr(self.spec, "eline_broad", False)
            if not spec_has_broad:
                import warnings

                warnings.warn(
                    "Spectroscopy has eline_broad=True but Parameters was built with "
                    "eline_broad=False. The broad-component velocity dispersion parameter "
                    "'eline_broad_sigma_kms' will not be sampled. "
                    "Pass eline_broad=True to Parameters() to fix this.",
                    UserWarning,
                    stacklevel=2,
                )

    def _init_eline_arrays(self, _spec_config):
        """Build catalog arrays and constraint matrices for emission line fitting."""
        from tengri.observation.line_list import LineList

        if _spec_config is not None and _spec_config.eline_catalog is not None:
            _catalog = _spec_config.effective_catalog
        else:
            _catalog = LineList.default_13()

        self._eline_wavelengths = _catalog.wavelengths
        self._eline_independent_wavelengths = _catalog.independent_wavelengths
        self._eline_names = _catalog.names

        fix_doublets = True
        if _spec_config is not None and hasattr(_spec_config, "eline_fix_doublets"):
            fix_doublets = _spec_config.eline_fix_doublets
        if fix_doublets:
            self._eline_constraint_matrix = _catalog.build_constraint_matrix()
        else:
            self._eline_constraint_matrix = jnp.eye(_catalog.n_lines)

        self._eline_prior_sigma = (
            getattr(_spec_config, "eline_prior_sigma", 100.0) if _spec_config else 100.0
        )
        self._eline_prior_width_dex = (
            getattr(_spec_config, "eline_prior_width_dex", 0.3) if _spec_config else 0.3
        )

        if self._eline_fitted:
            _secondary_indices = {dc.secondary_idx for dc in _catalog.doublets}
            _independent_line_names = [
                nm for i, nm in enumerate(_catalog.names) if i not in _secondary_indices
            ]
            self._eline_amplitude_names = [f"eline_amp_{nm}" for nm in _independent_line_names]
            _amp_bound = 10.0 * self._eline_prior_sigma
            _amp_priors = {
                nm: Uniform(-_amp_bound, _amp_bound) for nm in self._eline_amplitude_names
            }
            # Retained so the merge can be re-applied after the fit-time
            # approx policy reassigns ``self.spec`` (see __init__): that
            # reassignment drops params merged here otherwise.
            self._eline_amp_priors = _amp_priors
            self.spec = self.spec.merge_observation_params(**_amp_priors)
        else:
            self._eline_amplitude_names = []
            self._eline_amp_priors = {}

    def _build_data_args(self, model: Any) -> dict:
        """Build the data-dependent argument dict passed to JIT'd loss functions.

        These are passed as explicit arguments (not closed over) so that
        engines compiled for one galaxy can be reused for another with
        the same model + parameter structure.

        2026-05-23 (issue #250 follow-up): also threads the
        big template arrays (SSP grid, nebular templates, dust IR / AGN
        template data, fixed-value dict) so the **outer** JIT used by
        loss-fn-based samplers (HMC, NUTS, raytrace) sees them as
        Parameters, not Constants. Without this, the outer JIT inlines
        ``model.predict_observables_jit(params)`` and bakes the SSP
        flux grid (15×93×5994 floats) into the HLO as a constant —
        ballooning compile time from <5 s to 40 s on photometry.
        """
        args = {
            "data": self.data,
            "noise": self.noise,
            "sqrt_noise_inv": inv_noise_std(self.noise),
            "n_data": jnp.int32(len(self.data)),
        }
        if self.data_mask is not None:
            args["data_mask"] = self.data_mask
        if self.presence is not None:
            args["presence"] = self.presence
        if self._runtime_redshift is not None:
            # Runtime-routed redshift override (#1316): the loss replaces the
            # baked fixed value with this traced input (#1349's injection).
            args["redshift"] = jnp.asarray(self._runtime_redshift)

        obs = getattr(model, "observation", None)
        if obs is not None:
            spec_cfg = getattr(obs, "spectroscopy", None)
            if spec_cfg is not None and getattr(spec_cfg, "has_covariance", False):
                args["spec_cov_inv"] = spec_cfg.cov_inv

            line_flux_cfg = self._resolved_line_fluxes()
            if line_flux_cfg is not None:
                args["line_flux_obs"] = line_flux_cfg.fluxes
                args["line_flux_err"] = line_flux_cfg.errors
                args["line_flux_waves"] = line_flux_cfg.wavelengths
                limit_mask = getattr(line_flux_cfg, "limit_mask", None)
                if limit_mask is not None:
                    args["line_flux_limit_mask"] = limit_mask

            line_ratio_cfg = getattr(obs, "line_ratios", None)
            if line_ratio_cfg is not None:
                # The model ratio is computed from model.observation.line_ratios
                # in the loss closure; only the observed ratio + error need to
                # ride along for the Gaussian comparison.
                args["line_ratio_obs"] = line_ratio_cfg.ratios
                args["line_ratio_err"] = line_ratio_cfg.errors

            index_cfg = getattr(obs, "spectral_indices", None)
            if index_cfg is not None:
                args["index_obs"] = index_cfg.values
                args["index_err"] = index_cfg.errors

        # Outer-JIT threading: big arrays go in here so loss-fn callers
        # (HMC/NUTS) see them as outer Parameters, not Constants. Stored
        # under a private "_jit_inputs" sub-dict so existing data_args
        # consumers don't have to skip new keys.
        # Some test/dummy models don't implement the threading API. Decide that by
        # ASKING (hasattr) rather than by catching AttributeError out of the body:
        # a blanket ``suppress(AttributeError, TypeError)`` around the whole block
        # also swallows an AttributeError raised *from inside* a real model, and
        # then silently ships an un-threaded fit. That is exactly what happened —
        # ``ForwardModel`` (the canonical surface) did not delegate ``ssp_data``, so
        # every fit through it baked the SSP grid into the compiled program as a
        # constant and XLA was OOM-killed on large grids. A guard that fails open
        # turns a one-line omission into an invisible performance cliff.
        #
        # The topology gate is separate and asked of the model: the threaded
        # forward is written for a single-population SED forward and mis-broadcasts
        # the galaxy axis on a hierarchical one. Absent on SEDModel, where threading
        # has always been valid, so default True.
        #
        # Consequence worth stating: excluded topologies fall back to the eager
        # ``_build_prediction`` path, which closure-captures the SSP grid — so
        # hierarchical fits keep paying the baking cost #1496 removed for
        # single-galaxy ones (measured 5.7 h / 6.25 GB for a joint NUTS at N=4,
        # D=98). Fixing that needs the batched forward (#211), not a wider gate.
        _supports = getattr(model, "_supports_jit_threading", None)
        _threadable = _supports() if callable(_supports) else True
        if _threadable and all(
            hasattr(model, attr) for attr in ("spec", "ssp_data", "_template_data_for_jit")
        ):
            # Per-fit params override (#1329): the forward pass reads fixed values
            # (e.g. redshift under ``catalog_z_range``) from this threaded dict at
            # runtime, so the override MUST be merged here — not only in
            # ``_to_physical`` (the output-conversion path). Merging only there is a
            # silent relabel: the loss still runs at the model's fixed value while the
            # returned ``params`` echo the override. Keys are already validated in
            # ``__init__`` (fixed-only, real names).
            jit_fixed_values = dict(model.spec.get_fixed_values())
            if self._params_override is not None:
                jit_fixed_values.update(self._params_override)
            args["_jit_inputs"] = {
                "fixed_values": jit_fixed_values,
                "ssp_data": model.ssp_data,
                "template_data": model._template_data_for_jit(),
            }

        return args

    # ── Compilation ───────────────────────────────────────────────────

    def _resolve_compile_modes(
        self, compile_modes: tuple[str, ...] | str | None
    ) -> tuple[str, ...]:
        """Normalize compile_modes parameter to a tuple.

        Parameters
        ----------
        compile_modes : tuple[str, ...] or str or None
            User-provided compile modes specification.

        Returns
        -------
        tuple[str, ...]
            Normalized modes. Empty tuple means skip background compile.

        Notes
        -----

        - ``None`` → ``()`` (no background compile)
        - ``"auto"`` → infer from ``spec.stochastic`` and ``data_type``
        - ``str`` → wrap as ``(str,)``
        - ``tuple`` → return as-is

        """
        if compile_modes is None:
            return ()

        if isinstance(compile_modes, str):
            if compile_modes == "auto":
                return self._infer_default_compile_modes()
            return (compile_modes,)

        if isinstance(compile_modes, tuple):
            return compile_modes

        raise TypeError(
            f"compile_modes must be None, str, or tuple[str, ...]; "
            f"got {type(compile_modes).__name__}"
        )

    def _infer_default_compile_modes(self) -> tuple[str, ...]:
        """Infer sensible compile modes from model and data configuration.

        Returns
        -------
        tuple[str, ...]
            Recommended modes: VI for stochastic SFH, NUTS for parametric.
        """
        if self.spec.stochastic:
            return ("linear_resample", "nonlinear_update")

        if self.data_type == "photometry":
            return ("mcmc_nuts",)

        return ("mcmc_nuts",)

    def _start_background_compilation(self) -> None:
        """Spawn a daemon thread to pre-compile the JIT engine (if enabled).

        Background compilation is controlled by the ``compile_modes`` parameter
        passed to ``__init__``. If ``_target_modes`` is empty or the environment
        variable ``TENGRI_NO_BACKGROUND_COMPILE`` is set, no thread is spawned
        and ``_compilation_event`` is set immediately.

        When enabled, XLA C++ compilation releases the GIL, so this runs in
        genuine parallel with the caller's Python setup code. The
        ``_compilation_event`` is set before the first ``run()`` call can
        proceed past ``_get_or_build_engine``.
        """
        import os

        if os.environ.get("TENGRI_NO_BACKGROUND_COMPILE") or not self._target_modes:
            self._compilation_event.set()
            return

        def _worker() -> None:
            """Background thread that compiles the JIT engine."""
            try:
                with self._compilation_lock:
                    from tengri.inference.jit_engine import _SHARED_ENGINE_CACHE

                    sig = self.compile_signature()
                    if sig not in _SHARED_ENGINE_CACHE:
                        self.compile(
                            modes=self._target_modes,
                            verbose=False,
                        )
            except Exception as exc:
                logger.error(
                    "Background JIT compilation failed: %s: %s",
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                self._compilation_error = exc
            finally:
                self._compilation_event.set()

        thread = threading.Thread(target=_worker, daemon=True)
        self._compilation_thread = thread
        thread.start()

    def compile_signature(self) -> tuple:
        """Return a hashable signature for cross-galaxy engine reuse.

        Combines SEDModel's compile_signature() with Fitter-specific
        parameters that affect the compiled inference engine. Two Fitters
        with matching signatures can share the same XLA-compiled engine,
        even if they reference different SEDModel instances (as long as
        those instances have the same compile_signature).

        The signature does NOT include memory_mode, as it does not change
        the generated HLO graph — it only affects posterior-chunking
        behavior in the analysis layer (see _draw_jit_samples and
        _draw_nonlinear_jit_samples). Toggling memory_mode between
        "fast" and "low" reuses the same cached engine.

        Returns
        -------
        tuple
            Hashable immutable signature suitable for keying into
            the module-level _SHARED_ENGINE_CACHE.

        Notes
        -----
        Used by _get_or_build_engine to enable cross-galaxy engine reuse
        in PopulationFitter and CatalogFitter. The signature is computed
        ONCE per Fitter construction and cached to avoid recomputation
        in tight loops.
        """
        model_sig = self.model.compile_signature()
        # Single source of truth with the engine cache: _engine_cache_key
        # carries the observation feature channels (line fluxes / ratios /
        # indices / censoring mask) whose presence is baked into the loss
        # closure. Keeping them out of this signature lets a joint
        # phot+lines Fitter silently reuse a photometry-only loss (line
        # term dropped) or crash on a missing data_args key.
        return (model_sig, self._engine_cache_key())

    @property
    def _lean_keep_sig(self) -> tuple:
        """Cache-key signature that smart-lean preserves across runs.

        Single source of truth for the shape contract between
        ``Fitter.run(lean=True)`` and ``_SHARED_ENGINE_CACHE``: both
        sides must use this exact tuple, otherwise smart-lean drops
        the entry it was supposed to keep and every run recompiles.
        Pinned by ``test_lean_keep_sig_matches_engine_cache_key``.
        """
        return self.compile_signature()

    def _engine_cache_key(self) -> tuple:
        """Return a hashable key identifying the JIT engine shape.

        Two Fitters sharing the same Model will reuse the same compiled
        engine if their cache keys match (same data_type, stochastic
        flag, latent dimension, data length, free parameter names, noise
        model presence, and observation feature channels).

        The feature-channel entries (line fluxes / line ratios / spectral
        indices / censoring mask) are load-bearing: the loss closure bakes
        ``has_line_fluxes`` etc. in at build time, so two Fitters that
        differ only in these channels produce *different* loss functions.
        Without them in the key, a joint phot+lines fit silently reuses a
        photometry-only engine and drops the line term from the
        likelihood (or crashes with a missing ``line_flux_waves`` key,
        depending on build order).
        """
        from tengri.observation.noise import has_noise_model

        obs = getattr(self.model, "observation", None)
        line_flux_cfg = self._resolved_line_fluxes()
        line_flux_key = (
            (
                tuple(round(float(w), 6) for w in np.asarray(line_flux_cfg.wavelengths)),
                # Limit-mask PRESENCE selects Censored vs Gaussian adapters
                # (structure); the mask VALUES ride through data_args.
                getattr(line_flux_cfg, "limit_mask", None) is not None,
            )
            if line_flux_cfg is not None
            else None
        )
        line_ratio_cfg = getattr(obs, "line_ratios", None) if obs is not None else None
        index_cfg = getattr(obs, "spectral_indices", None) if obs is not None else None

        return (
            self.data_type,
            self.spec.stochastic,
            self.spec.n_grid if self.spec.stochastic else 0,
            len(self.data),
            tuple(sorted(self._free_names)),
            has_noise_model(self.spec),
            self._eline_marginalize,
            self._eline_fitted,
            self._calibration_marginalize,
            self._eline_prior_type,
            line_flux_key,
            line_ratio_cfg is not None,
            index_cfg is not None,
            self.data_mask is not None,
            # Per-fit params override (#1329): the loss closure bakes
            # ``fitter._fixed_values``, which now carries the override, so two
            # fits differing only by override MUST get distinct loss functions —
            # exactly like the feature channels above. Without this, fit #2
            # silently reuses fit #1's baked override.
            #
            # EXCEPT a runtime-routed redshift (#1316): it rides ``data_args``
            # as a traced input, so distinct z legitimately share one program.
            # Note a routed-z-only override yields ``()``, distinct from the
            # no-override ``None`` — a plain fit (no data_args redshift) never
            # shares a closure whose baked z differs from its spec.
            (
                tuple(
                    sorted(
                        (k, round(float(v), 8))
                        for k, v in self._params_override.items()
                        if not (k == "redshift" and self._runtime_redshift is not None)
                    )
                )
                if self._params_override
                else None
            ),
            # Free-parameter PRIOR identity. ``_build_signal_response`` threads
            # only data/noise through ``data_args``; the priors stay baked,
            # because ``_primals_to_params`` calls ``dist.unstandardize(xi)``
            # and that reads the distribution's Python floats at trace time
            # (``Uniform``: ``lo + (hi - lo) * Phi(xi)``). Baked is fine — but
            # only if keyed. Without this entry two models differing solely in a
            # prior's bounds share one engine, and fit #2's latent is decoded
            # through fit #1's interval: a shift of order the prior width, which
            # on ``log_total_mass`` reads as a mass deviation of order dex.
            # Exactly the ``params_override`` hazard above, one layer over.
            # Free NAMES (field 5) cannot stand in for this: editing
            # ``Uniform(9.6, 11.1)`` to ``Uniform(7, 13)`` changes no name, no
            # shape, no dtype and no control flow.
            self._free_prior_key(),
            # Spec FIXED values (#1972 instance 2). ``_primals_to_params`` also
            # bakes ``fitter._fixed_values``, so two models differing only in a
            # fixed scalar share one engine and fit #2 runs fit #1's physics —
            # measured -0.18 dex on mass for ``dust_slope`` -0.7 -> 0.4.
            #
            # ``SEDModel.compile_signature`` dropped these on 2026-05-20 on the
            # grounds that they "are threaded as a runtime JIT input"; that is
            # true of the forward observables path and false of this closure, so
            # do not take that comment as cover for removing this entry.
            #
            # Keying rather than threading is deliberate:
            # ``get_or_build_signal_response`` returns a *stable function
            # object* because JAX's trace cache is keyed by function identity,
            # so partial-applying fixed values per fit would re-trace the whole
            # physics stack per galaxy — the cost that cache exists to avoid.
            # The keying is free for catalogs: these are per-MODEL values and a
            # catalog uses one model, with per-galaxy variation flowing through
            # ``_params_override`` (keyed above) or a runtime-routed redshift.
            self._fixed_value_key(),
            # Mirror map (#1972 instance 3). ``_primals_to_params`` calls
            # ``spec.resolve_mirrors``, baking target -> source. Two specs can
            # share every free name, every fixed name and every prior while
            # tying the same target to a DIFFERENT source; without this entry
            # the second silently ties to the first's source.
            self._mirror_key(),
        )

    def _free_prior_key(self) -> tuple:
        """Return the prior identity of every free parameter, in sorted order.

        Delegates per-distribution to
        :meth:`~tengri.parameters.priors.Distribution.jit_cache_key`, which is
        exclusion-based so a new distribution family is keyed correctly without
        touching this method.

        Fixed parameters are deliberately absent: their values already ride
        field 5's companion ``spec_fixed_id`` in
        :meth:`SEDModel.compile_signature` and the ``params_override`` entry
        above, and a ``Fixed`` prior contributes no transform to bake.

        A prior that does not inherit from ``Distribution`` (a caller's own
        class) falls back to ``repr``. That fails in the safe direction: a
        default ``repr`` carries the instance address, so two such priors never
        share an engine. The cost is a missed cache hit, not a wrong decode.
        """
        return tuple(
            (
                name,
                (
                    dist.jit_cache_key()
                    if hasattr(dist, "jit_cache_key")
                    else repr(dist)  # unknown prior type: never share (see above)
                ),
            )
            for name, dist in (
                (n, self.spec.get_distribution(n)) for n in sorted(self._free_names)
            )
        )

    def _fixed_value_key(self) -> tuple:
        """Return the spec's fixed parameter VALUES, in sorted order.

        The per-fit override is keyed separately (the ``params_override`` entry),
        and a runtime-routed redshift is excluded for the same reason it is
        excluded there: under ``catalog_z_range`` it rides ``data_args`` as a
        traced input (#1316), so distinct redshifts legitimately share one
        program and keying it would recompile per galaxy.

        Values are read from ``self.spec`` rather than ``self._fixed_values``
        because the latter already has the override merged in, which would
        double-count it and reintroduce the per-galaxy recompile.
        """
        from tengri.parameters.priors import hashable_baked_value

        fixed = self.spec.get_fixed_values()
        return tuple(
            (name, hashable_baked_value(fixed[name]))
            for name in sorted(fixed)
            if not (name == "redshift" and self._runtime_redshift is not None)
        )

    def _mirror_key(self) -> tuple:
        """Return the mirror map (target -> source), in sorted order.

        ``_primals_to_params`` calls ``spec.resolve_mirrors``, so the map is a
        baked constant. Only the *map* is keyed; the mirrored values themselves
        arrive from the latents and need no key.
        """
        mirrors = getattr(self.spec, "mirrors", None) or {}
        return tuple(sorted((str(target), str(source)) for target, source in mirrors.items()))

    def _get_or_build_engine(self, pos_dict: dict) -> dict:
        """Return the JIT engine, reusing a cached version when possible.

        Engines are cached in a module-level shared cache keyed by
        compile_signature(), enabling zero-recompile fits when multiple
        Fitters share the same model structure (e.g., catalog fits with
        different SSP files of identical shape).

        Also maintains a backward-compat per-model cache for any code
        that reads the model's cache namespace under ``"jit_engine"``.

        Blocks until the background compilation thread (started in
        ``__init__``) has finished.  On an XLA cache hit the wait is
        effectively instant (<1 s).
        """
        # Skip the wait if called from the background compilation thread
        # itself (via compile() → _get_or_build_engine) to avoid deadlock.
        if threading.current_thread() is not self._compilation_thread:
            self._compilation_event.wait()
            if self._compilation_error is not None:
                raise RuntimeError(
                    "Background JIT compilation failed."
                ) from self._compilation_error

        if self._jit_sampler is not None:
            return self._jit_sampler

        # Look up in shared cross-galaxy cache first
        from tengri.inference.jit_engine import get_or_build_engine_cached

        engine = get_or_build_engine_cached(self, pos_dict)

        # Write-through to per-model cache for backward compat
        cache_key = (self._engine_cache_key(), getattr(self, "_memory_mode", "fast"))
        per_model_cache = _model_cache_owner.get_or_compile_model(self.model).setdefault(
            "jit_engine", {}
        )
        per_model_cache[cache_key] = engine

        self._jit_sampler = engine
        return engine

    def _build_jit_engine(self, pos_dict):
        """Build JIT-compiled inference engine. See ``jit_engine.build_jit_engine``."""
        return build_jit_engine(self, pos_dict)

    def compile(
        self,
        *,
        n_iterations=15,
        n_samples=3,
        n_posterior_samples=200,
        modes=("linear_resample", "nonlinear_update"),
        mcmc_methods=(),
        n_warmup=300,
        n_burnin=100,
        n_mcmc_samples=100,
        nss=False,
        verbose=True,
    ):
        """Pre-compile the JIT inference engine ahead of time.

        Triggers XLA compilation for all specified modes so that
        subsequent ``fitter.run()`` calls have zero compilation delay.
        Compiled programs are cached both in-memory (this session)
        and on disk (``/tmp/tengri_jax_cache``, survives restarts).

        Parameters
        ----------
        n_iterations : int
            Iteration count for the pre-compilation run.  Changing
            ``n_iterations`` at run time does NOT trigger recompilation
            (the iteration count is a dynamic traced value).
        n_samples : int
            Compile for this sample count.  Changing ``n_samples``
            at run time DOES trigger recompilation (array shapes
            depend on it).
        n_posterior_samples : int
            Compile posterior draw for this many samples.
        modes : tuple of str
            Which VI sample modes to pre-compile. Each mode compiles
            separately. Default covers MGVI + geoVI update (fastest).
            Add ``"nonlinear_resample"`` for full geoVI (~56s extra).
        mcmc_methods : tuple of str
            MCMC methods to pre-compile. Supported values:
            ``"nuts"``, ``"hmc"``, ``"dynamic_hmc"``, ``"ghmc"``.
            Each call runs the full warmup + chain scan through JIT so
            the XLA disk cache is populated before the first user call.
            After ``fitter.compile(mcmc_methods=["nuts"])``, a fresh
            kernel restart deserializes in <1s instead of ~23s.
        n_warmup : int
            Warmup steps used for the MCMC compilation run.
        n_burnin : int
            Burn-in steps used for the MCMC compilation run.
        n_mcmc_samples : int
            Sample steps used for the MCMC compilation run.
        nss : bool
            Pre-compile the NSS (nested slice sampling) step and init
            functions.  NSS has a ~10–15s cold compile on the first
            ``fitter.run("nss")`` call; setting ``nss=True`` moves that cost
            to compile time.  ``data_args`` is traced so the compiled program
            is reused across galaxies with the same model configuration.
        verbose : bool
            Print compilation progress.

        Returns
        -------
        self

        Notes
        -----
        **Compilation mechanics**: Pre-compilation invokes ``jax.jit`` on
        the forward model's SED prediction and inference engines, storing
        compiled XLA programs to disk. First ``fitter.run()`` will skip
        XLA overhead by loading pre-compiled kernels. Typical times:
        ``"linear_resample"`` + ``"nonlinear_update"`` ~3s; full modes ~60s;
        NUTS ~23s (once per unique model shape).

        **MCMC cache key**: The XLA program is keyed on ``logdensity_fn_2arg``
        identity, ``n_warmup``, ``n_burnin``, ``n_mcmc_samples``, and
        ``use_dense``.  Use the same values here as in ``fitter.run()`` to
        guarantee a cache hit.  Changing galaxy data does **not** invalidate
        the cache (``data_args`` is traced, not static).

        **JIT-compatible**: yes — internally calls JIT-compiled JAX functions.

        Example
        -------
        >>> fitter = Fitter(forward, data, noise)
        >>> fitter.compile()  # ~3s for default VI modes
        >>> fitter.compile(mcmc_methods=["nuts"])  # ~23s, then instant restarts
        >>> fitter.compile(nss=True)  # ~12s, then instant restarts
        >>> result = fitter.run("mcmc_nuts")  # instant after compile
        >>> result = fitter.run("nss")  # instant after compile
        """
        dummy_pos = self._initialize_unbounded(jax.random.PRNGKey(0))
        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(dummy_pos)

        engine = self._jit_sampler
        flatten = engine["flatten"]
        pos_flat = flatten(dummy_pos)
        data_args = self._data_args

        if verbose:
            logger.info(
                "Compiling: n_iter=%d, n_samp=%d, n_post=%d, modes=%s",
                n_iterations,
                n_samples,
                n_posterior_samples,
                modes,
            )

        # Pre-compile each optimization mode
        for mode in modes:
            if verbose:
                logger.info("  Compiling %s...", mode)
            t0 = time.time()
            engine["run_evi_geovi"](
                pos_flat,
                jax.random.PRNGKey(0),
                data_args,
                n_iterations=n_iterations,
                n_samples=n_samples,
                kl_rtol=0.0,
                sample_mode=mode,
            )
            if verbose:
                logger.info("  Compiling %s... %.1fs", mode, time.time() - t0)

        # Pre-compile MGVI optimizer (old path, used by native_mgvi)
        if verbose:
            logger.info("  Compiling MGVI (old path)...")
        t0 = time.time()
        engine["run_evi"](
            pos_flat,
            jax.random.PRNGKey(0),
            data_args,
            n_iterations=n_iterations,
            n_samples=n_samples,
            kl_rtol=1e-2,
        )
        if verbose:
            logger.info("  Compiling MGVI (old path)... %.1fs", time.time() - t0)

        # Pre-compile posterior draw
        if verbose:
            logger.info(
                "  Compiling posterior draw (%d samples)...",
                n_posterior_samples,
            )
        t0 = time.time()
        draw_keys = jax.random.split(jax.random.PRNGKey(0), n_posterior_samples)
        engine["draw_samples"](pos_flat, draw_keys, data_args)
        if verbose:
            logger.info(
                "  Compiling posterior draw (%d samples)... %.1fs",
                n_posterior_samples,
                time.time() - t0,
            )

        if mcmc_methods:
            from tengri.inference.backends.mcmc._shared import (
                DEFAULT_MAX_NUM_DOUBLINGS,
                _dynamic_hmc_full_scan,
                _get_flat_logdensity,
                _ghmc_full_scan,
                _hmc_full_scan,
                _nuts_full_scan,
            )

            log_posterior_flat_2arg, _, init_flat, data_args = _get_flat_logdensity(
                self, dummy_pos
            )
            n_chain = n_burnin + n_mcmc_samples
            warmup_key = jax.random.PRNGKey(1)
            chain_keys = jax.random.split(jax.random.PRNGKey(2), n_chain)
            n_dim = len(init_flat)
            use_dense = n_dim <= 30

            for method in mcmc_methods:
                if verbose:
                    logger.info("  Compiling MCMC %s...", method)
                t0 = time.time()
                if method in ("nuts", "mcmc_nuts"):
                    _nuts_full_scan(
                        init_flat,
                        warmup_key,
                        chain_keys,
                        log_posterior_flat_2arg,
                        data_args,
                        n_warmup,
                        DEFAULT_MAX_NUM_DOUBLINGS,
                        use_dense,
                        0.85,
                    )
                elif method in ("hmc", "mcmc_hmc"):
                    _hmc_full_scan(
                        init_flat,
                        warmup_key,
                        chain_keys,
                        log_posterior_flat_2arg,
                        data_args,
                        n_warmup,
                        10,
                        use_dense,
                        0.85,
                    )
                elif method in ("dynamic_hmc", "mcmc_dynamic_hmc"):
                    dhmc_init_key = jax.random.PRNGKey(4)
                    dhmc_chain_keys = jax.random.split(jax.random.PRNGKey(5), n_chain)
                    _dynamic_hmc_full_scan(
                        init_flat,
                        warmup_key,
                        dhmc_init_key,
                        dhmc_chain_keys,
                        log_posterior_flat_2arg,
                        data_args,
                        n_warmup,
                        use_dense,
                        0.85,
                    )
                elif method in ("ghmc", "mcmc_ghmc"):
                    ghmc_init_key = jax.random.PRNGKey(4)
                    ghmc_chain_keys = jax.random.split(jax.random.PRNGKey(5), n_chain)
                    _ghmc_full_scan(
                        init_flat,
                        warmup_key,
                        ghmc_init_key,
                        ghmc_chain_keys,
                        log_posterior_flat_2arg,
                        data_args,
                        n_warmup,
                        0.85,
                        0.8,
                        0.65,
                    )
                else:
                    logger.warning("  Unknown MCMC method for compile: %s", method)
                    continue
                if verbose:
                    logger.info("  Compiling MCMC %s... %.1fs", method, time.time() - t0)

        if nss:
            from tengri.inference.backends.evidence import _get_nss_fns

            if self.spec.stochastic:
                logger.warning("  NSS compile skipped: NSS does not support stochastic SFH")
            else:
                D = len(self._free_names)
                if verbose:
                    logger.info("  Compiling NSS (D=%d)...", D)
                t0 = time.time()
                init_jit, step_jit = _get_nss_fns(
                    self,
                    num_inner_steps=D,
                    num_delete=50,
                    max_steps=10,
                    max_shrinkage=100,
                )
                nss_key = jax.random.PRNGKey(10)
                nss_key, init_key = jax.random.split(nss_key)
                all_samples = self.spec.sample_batch(init_key, 200)
                particles = {name: all_samples[name] for name in self._free_names}
                live = init_jit(particles, data_args)
                nss_key, step_key = jax.random.split(nss_key)
                step_jit(step_key, live, data_args)
                if verbose:
                    logger.info("  Compiling NSS... %.1fs", time.time() - t0)

        if verbose:
            logger.info("Compilation complete.")
        return self

    # ── Loss and likelihood builders ──────────────────────────────────

    def _build_loss_fn(self) -> Callable:
        """Build a differentiable loss function.

        See ``tengri.inference.loss_functions.build_loss_fn`` for full docs.
        Returns ``loss_fn(params_unbounded, data_args) -> scalar``.
        """
        return build_loss_fn(self)

    def _get_or_build_loss_fn(self) -> Callable:
        """Return the cached, JIT-compiled loss function, building if needed.

        Cached on the Model object keyed by ``_engine_cache_key()`` so
        multiple Fitters with the same model structure share one
        compiled XLA program.

        The built objective is wrapped in :func:`jax.jit` before caching so
        the callable exposed through
        :attr:`tengri.inference.context.InferenceContext.neg_log_posterior_fn`
        is genuinely JIT-cached, as its docstring and ADR-0010 promise —
        rather than a Python-level orchestration of the per-component chain.
        Without the wrapper a raw ``neg_log_posterior_fn(params, data_args)``
        call re-runs the chain dispatcher at Python level every evaluation:
        ~20x slower on joint fits that add a spectral-index / line-flux
        channel, where the feature forward (``predict_state``) is otherwise
        never fused. Gradient-based backends were already protected —
        ``grad_fn`` / ``logdensity_fn`` wrap ``value_and_grad`` in their own
        ``jax.jit`` — so this only closes the gap for the objective itself
        (VI, nested sampling, custom loops, MAP convergence logging).
        ``jax.value_and_grad`` differentiates transparently through the inner
        ``jax.jit`` (``grad(jit(f)) == grad(f)``), and XLA subsumes the nested
        trace, so those already-JIT'd callers see no runtime cost.
        """
        from tengri.inference.jit_engine import get_or_build_cached

        cache_key = self._engine_cache_key()
        per_model = _model_cache_owner.get_or_compile_model(self.model).setdefault("loss_fn", {})
        if cache_key in per_model:
            return per_model[cache_key]
        loss_fn = jax.jit(get_or_build_cached(self, "loss", self._build_loss_fn))
        per_model[cache_key] = loss_fn
        return loss_fn

    def _build_logprior_fn(self) -> Callable:
        """Build a log-prior function. See ``loss_functions.build_logprior_fn``."""
        return build_logprior_fn(self)

    def _build_loglikelihood_fn(self) -> Callable:
        """Build log-likelihood function. See ``loss_functions.build_loglikelihood_fn``."""
        return build_loglikelihood_fn(self)

    def _get_or_build_loglikelihood_fn(self) -> Callable:
        """Return the cached log-likelihood function, building if needed."""
        from tengri.inference.jit_engine import get_or_build_cached

        cache_key = self._engine_cache_key()
        per_model = _model_cache_owner.get_or_compile_model(self.model).setdefault("loglik_fn", {})
        if cache_key in per_model:
            return per_model[cache_key]
        loglik_fn = get_or_build_cached(self, "loglik", self._build_loglikelihood_fn)
        per_model[cache_key] = loglik_fn
        return loglik_fn

    def _build_loglikelihood_unbounded_fn(self) -> Callable:
        """Build unbounded-space log-likelihood.

        See ``loss_functions.build_loglikelihood_unbounded_fn``.
        """
        return build_loglikelihood_unbounded_fn(self)

    def _get_or_build_loglikelihood_unbounded_fn(self) -> Callable:
        """Return the cached unbounded-space log-likelihood, building if needed.

        Caches per-model (no shared cross-fitter cache, unlike ``loss_fn``);
        unbounded-space log-likelihood is a thin wrapper over the data term
        plus :func:`_unstandardize_parameters`, so the compile cost is
        marginal and the cache key would mirror the loss cache anyway.
        """
        cache_key = self._engine_cache_key()
        per_model = _model_cache_owner.get_or_compile_model(self.model).setdefault(
            "loglik_unbounded_fn", {}
        )
        if cache_key in per_model:
            return per_model[cache_key]
        fn = self._build_loglikelihood_unbounded_fn()
        per_model[cache_key] = fn
        return fn

    def _get_or_build_grad_fn(self) -> Callable:
        """Return cached JIT-compiled value_and_grad of the loss function.

        The gradient function takes ``(params_unbounded, data_args)`` as
        explicit arguments so the compiled XLA program is reusable across
        galaxies with the same model structure.
        """
        from tengri.inference.jit_engine import get_or_build_cached

        cache_key = self._engine_cache_key()
        per_model = _model_cache_owner.get_or_compile_model(self.model).setdefault("grad_fn", {})
        if cache_key in per_model:
            return per_model[cache_key]

        loss_fn = self._get_or_build_loss_fn()

        def _build():
            @jax.jit
            def val_and_grad(params_u, data_args):
                """Loss and gradient w.r.t. unbounded parameters."""
                return jax.value_and_grad(lambda p: loss_fn(p, data_args))(params_u)

            return val_and_grad

        val_and_grad = get_or_build_cached(self, "grad", _build)
        per_model[cache_key] = val_and_grad
        return val_and_grad

    def _get_or_build_logdensity_fn(self) -> Callable:
        """Return cached JIT-compiled log-density for MCMC/Pathfinder.

        Returns ``logdensity(params_u, data_args) -> scalar``.  Callers
        should partial-apply ``data_args`` for blackjax compatibility.
        """
        cache_key = self._engine_cache_key()
        from tengri.inference.jit_engine import get_or_build_cached

        per_model = _model_cache_owner.get_or_compile_model(self.model).setdefault(
            "logdensity_fn", {}
        )
        if cache_key in per_model:
            return per_model[cache_key]

        loss_fn = self._get_or_build_loss_fn()

        def _build():
            @jax.jit
            def logdensity(params_u, data_args):
                """Log posterior (negative loss) for MCMC."""
                return -loss_fn(params_u, data_args)

            return logdensity

        logdensity = get_or_build_cached(self, "logdensity", _build)
        per_model[cache_key] = logdensity
        return logdensity

    # ── Parameter transforms ──────────────────────────────────────────

    def _initialize_unbounded(self, key: Any) -> dict:
        """Create initial unbounded parameter dict.

        Per-param shape comes from ``self.spec.param_init_shape(name)``
        when available (the PopulationSpecView publishes this — see
        PR #239 plan Task 5). Scalar specs (the standard
        :class:`Parameters`) fall back to shape ``()``.

        For hierarchical fits, per-galaxy free params get a leading
        ``(N,)`` axis; shared parameters stay scalar. This is what
        the standardized hierarchical Hamiltonian (paper §4) expects.
        """
        params = {}
        keys = jax.random.split(key, len(self._free_names) + 1)
        # Shape provider — defaults to scalar for non-Population specs
        get_shape = getattr(self.spec, "param_init_shape", lambda _n: ())

        for i, name in enumerate(self._free_names):
            dist = self.spec.get_distribution(name)
            shape = get_shape(name)
            if isinstance(dist, Gaussian):
                base = dist.standardize(jnp.array(dist.mu))
                params[name] = jnp.broadcast_to(base, shape) if shape else base
            else:
                # Initialize near midpoint (u=0) with small perturbation
                params[name] = 0.1 * jax.random.normal(keys[i], shape=shape)

        if self.spec.stochastic:
            # psd_xi shape: scalar fits use (n_grid,); hierarchical
            # populations use (N, n_grid) — published via the spec.
            psd_shape = getattr(self.spec, "psd_xi_init_shape", None) or (self.spec.n_grid,)
            # Property vs callable — both supported
            if callable(psd_shape):
                psd_shape = psd_shape()
            params["psd_xi"] = 0.1 * jax.random.normal(keys[-1], shape=psd_shape)

        return params

    def _unbounded_from_posterior(self, posterior: Posterior) -> dict:
        """Convert a Posterior's params to unbounded space for init."""
        params = {}
        for name in self._free_names:
            if name in posterior.params:
                dist = self.spec.get_distribution(name)
                params[name] = dist.standardize(jnp.array(posterior.params[name]))
            else:
                params[name] = jnp.array(0.0)

        if self.spec.stochastic and "psd_xi" in posterior.params:
            params["psd_xi"] = posterior.params["psd_xi"]
        elif self.spec.stochastic:
            params["psd_xi"] = jnp.zeros(self.spec.n_grid)

        return params

    def _to_physical(self, params_unbounded: dict) -> dict:
        """Convert a single unbounded param dict to physical space."""
        params = {}
        for name in self._free_names:
            dist = self.spec.get_distribution(name)
            params[name] = dist.unstandardize(params_unbounded[name])
        for name, val in self._fixed_values.items():
            # self._fixed_values already carries any per-fit params override
            # (#1329, merged at construction) — no separate merge needed here.
            params[name] = jnp.array(val)
        if self.spec.stochastic and "psd_xi" in params_unbounded:
            # Publish under both names so the returned ``Posterior.params``
            # evaluates to the model that was actually fitted: ``psd_xi`` is the
            # sampler's key, ``sfh_field_xi`` is the name the forward model and
            # the docs use. Emitting only ``psd_xi`` made
            # ``model.predict_photometry(posterior.params)`` silently score the
            # SMOOTH model -- chi2/N 0.34 read back as 9.00 (#1271).
            params["psd_xi"] = params_unbounded["psd_xi"]
            params["sfh_field_xi"] = params_unbounded["psd_xi"]
        return params

    # ── AOT pre-warm and adaptation persistence ──────────────────────

    def prewarm(self, method: str = "mcmc_nuts", *, n_chains: int | None = None, key=None):
        """Pre-compile JIT kernels and populate the adaptation cache for ``method``.

        After this returns, a subsequent :meth:`run` call with the same
        ``method`` skips XLA compilation **and** sampler warmup window
        adaptation — only the actual sampling work remains.

        Concretely: runs the smallest meaningful inference (a few
        warmup steps + a handful of samples) to (1) compile every
        JIT'd kernel in the loss / sampler stack against the current
        data shape, and (2) write step size + mass matrix (or
        equivalent) into the per-model adaptation cache. The
        persistent XLA cache (``~/.cache/tengri_jax_cache``) also
        captures the compile, so a fresh Python process sees a warm
        XLA cache too.

        Parameters
        ----------
        method : str, default ``"mcmc_nuts"``
            Inference method to pre-warm. Any name accepted by
            :meth:`run`.
        n_chains : int or None
            If set and greater than 1, also pre-compile the multichain
            ``jax.vmap`` path for ``n_chains`` so the second multichain
            call has zero compile latency. Only meaningful for backends
            that support ``n_chains`` (NUTS / HMC / dHMC / GHMC / MCLMC /
            adjusted MCLMC / raytrace).
        key : jax.random.PRNGKey or None
            Optional seed. Default uses a fixed key — the pre-warm run
            is throwaway and its randomness does not affect the
            subsequent real fit.

        Returns
        -------
        None

        Examples
        --------
        >>> from tengri.inference.fitter import Fitter
        >>> fitter = Fitter(forward, flux, noise, data_type="photometry")
        >>> fitter.prewarm(method="mcmc_nuts", n_chains=4)
        >>> posterior = fitter.run(method="mcmc_nuts", n_chains=4, n_samples=1000)

        Notes
        -----
        Pre-warming is **soft**: any exception raised during the throwaway
        call is swallowed so the real ``run()`` surfaces the genuine error
        with a richer traceback. Calling ``prewarm`` redundantly is cheap
        — both caches are short-circuited.
        """
        import jax as _jax

        if key is None:
            key = _jax.random.PRNGKey(0)
        sample_kwarg = "n_steps" if method in ("mcmc_raytrace", "raytrace") else "n_samples"
        warmup_kw = {sample_kwarg: 10}
        try:
            # prewarm=False: this throwaway run must not re-enter auto-prewarm
            # (run -> _auto_prewarm -> prewarm -> run would recurse).
            self.run(method=method, key=key, verbose=False, prewarm=False, **warmup_kw)
        except Exception:
            return
        if n_chains is not None and n_chains > 1:
            warmup_kw["n_chains"] = n_chains
            # Broad by design — this is a warmup, and any exception a real fit
            # can raise can surface here too, so narrowing the type would just
            # let some failures escape and abort a run that was going to work.
            # What was wrong is that the failure left no trace: a warmup that
            # silently stopped working looks exactly like one that ran, and the
            # only symptom is the compile cost reappearing in the real fit.
            try:
                self.run(method=method, key=key, verbose=False, prewarm=False, **warmup_kw)
            except Exception as exc:
                _prewarm_logger().debug(
                    "multi-chain warmup for method=%r did not complete (%s: %s); "
                    "the fit continues and will compile on first use",
                    method,
                    type(exc).__name__,
                    exc,
                )

    def _auto_prewarm(self, key) -> None:
        """JIT-compile the shared loss/grad + predict surface before the fit loop.

        Called by :meth:`run` when ``prewarm=True`` (the default). Compiles the
        two forward-model-heavy pieces every fit needs — the negative-log-
        posterior **gradient** (which every backend evaluates and which subsumes
        the loss compile) and the post-fit predict surface (``predict_photometry``
        + ``predict_properties`` on the fit model) — by building and evaluating
        each once, blocked. This populates the persistent JAX cache so the fit
        loop and immediate posterior-predictive / derived-quantity exploration
        run warm.

        Deliberately does **not** run a throwaway sampler fit: for optimizer
        backends (MAP) a throwaway ``run`` would repeat the full optimization,
        and the method-specific sampler-loop kernel is compiled once on the real
        run and persisted anyway. Use :meth:`prewarm` for explicit sampler AOT
        (e.g. per-``n_chains`` NUTS warmup).

        Best-effort: every step is wrapped so a failure here never masks the
        genuine error the real :meth:`run` would raise.
        """

        import jax as _jax

        if key is None:
            key = _jax.random.PRNGKey(0)
        # Loss + gradient: the forward-heavy compile shared by every backend.
        try:
            grad_fn = self._get_or_build_grad_fn()
            init = self._initialize_unbounded(key)
            _jax.block_until_ready(grad_fn(init, self._data_args))
        except Exception as exc:
            _prewarm_logger().debug(
                "gradient prewarm skipped (%s: %s); the fit continues and pays "
                "this compile on its first evaluation",
                type(exc).__name__,
                exc,
            )
        # Post-fit predict surface on the fit model (LUT-honoring accessors).
        # The wrappers are memoized per model: ``self.model.predict_photometry``
        # builds a NEW bound-method object on every attribute access, so a bare
        # ``jax.jit(...)`` here got a fresh cache entry and recompiled on every
        # fit — the warming step was the one thing that never stayed warm. It cost
        # two compiles per galaxy on a sequential catalog.
        try:
            warm_p = self.spec.sample(key)
            for _name in ("predict_photometry", "predict_properties"):
                _jax.block_until_ready(_memoized_predict_jit(self.model, _name)(warm_p))
        except Exception as exc:
            _prewarm_logger().debug(
                "predict-surface prewarm skipped (%s: %s); post-fit accessors "
                "will compile on first access",
                type(exc).__name__,
                exc,
            )

    def save_cache(self, path) -> None:
        """Persist this model's adaptation cache (step size + mass matrix) to disk.

        Reload in a fresh Python process with :meth:`load_cache` to skip
        sampler warmup on the next :meth:`run` call. Useful when the same
        model+data is fit repeatedly across notebook restarts or batch
        jobs — warmup window adaptation typically dominates first-call
        wall time on a warm XLA cache.

        Stored payload:

        - ``adaptation`` : dict keyed by ``(engine_key, method_key)`` — the
          contents of the model's cache namespace under ``"adaptation"``.
        - ``spec_fingerprint`` : a content hash of the free-parameter names
          and prior shape, used by :meth:`load_cache` to refuse to load a
          cache that was written for a different model.

        Parameters
        ----------
        path : str or Path
            Destination file (``.pkl`` recommended). Parent directory is
            created if missing.

        Returns
        -------
        None
        """
        import pickle
        from pathlib import Path as _Path

        mc = _model_cache_owner.get_or_compile_model(self.model)
        adaptation = mc.get("adaptation", {})
        # Cached MAP point estimate (if a previous .run() or .prewarm()
        # populated it). Saving it skips MAP init on the next load_cache
        # session — same speedup the adaptation cache gives for warmup.
        map_params = mc.get("map_params_physical")
        payload = {
            "adaptation": adaptation,
            "map_params_physical": map_params,
            "spec_fingerprint": self._spec_fingerprint(),
        }
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(payload, f)

    def load_cache(self, path) -> None:
        """Load an adaptation cache previously written by :meth:`save_cache`.

        Refuses to load if the spec fingerprint disagrees (different
        free-parameter set), to prevent silently using a stale cache.

        Parameters
        ----------
        path : str or Path
            File written by :meth:`save_cache`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the cache's ``spec_fingerprint`` does not match this
            Fitter's model.
        """
        import pickle
        from pathlib import Path as _Path

        # B301: `path` is a cache this process wrote via save_cache().
        # Unpickling executes arbitrary code, so the contract is that the
        # caller supplies their own file; the fingerprint check below is a
        # correctness guard against a *stale* cache, not a security boundary,
        # and it runs after the payload has already been deserialized.
        with _Path(path).open("rb") as f:
            payload = pickle.load(f)  # nosec B301
        if payload.get("spec_fingerprint") != self._spec_fingerprint():
            raise ValueError(
                f"Adaptation cache at {path} was written for a different model "
                "(spec fingerprint mismatch). Re-run prewarm + save_cache."
            )
        mc = _model_cache_owner.get_or_compile_model(self.model)
        mc.setdefault("adaptation", {}).update(payload["adaptation"])
        if payload.get("map_params_physical") is not None:
            mc["map_params_physical"] = payload["map_params_physical"]

    def _spec_fingerprint(self) -> str:
        """Content hash of free parameter names + ordered fixed values.

        Stable across processes; changes if the user reorders / renames
        parameters or changes which are fixed vs free. Does NOT depend on
        data tensors — the adaptation cache itself is data-conditional
        and the user takes responsibility for fitting the same model+data
        combination after :meth:`load_cache`.
        """
        import hashlib

        # A cache key over (free names, fixed values), not a security digest:
        # it decides whether a stored fit may be reused, and nothing trusts it
        # against a forged input. `usedforsecurity=False` says so to the reader,
        # to bandit (B324), and to FIPS builds, where a plain sha1() raises.
        h = hashlib.sha1(usedforsecurity=False)
        for name in self._free_names:
            h.update(name.encode())
        for name, val in sorted(self._fixed_values.items()):
            h.update(name.encode())
            h.update(f"{float(val):.10g}".encode())
        return h.hexdigest()

    # ── Inference dispatch ────────────────────────────────────────────

    def run(
        self,
        method: str = DEFAULT_METHOD,
        *,
        init_from=None,
        key=None,
        allow_unvalidated: bool = False,
        **kwargs,
    ):
        """Run inference using the specified method.

        Dispatches to the underlying inference backend (variational, MCMC,
        point estimation, or nested sampling) and returns a ``Posterior``
        object with samples, diagnostics, and derived quantities.

        Hierarchical fits (``model`` is a ForwardModel built with
        ``population=PopulationSEDModel(...)``) route through
        :class:`tengri.PopulationFitter` automatically. No change in
        the user-facing call site.

        Parameters
        ----------
        method : str, optional
            Inference method (case-sensitive). Default ``"vi"``.

            **Variational Inference (VI)**

            - ``"vi"``: geoVI via NIFTy (nonlinear, default for D>20)
            - ``"vi_nonlinear"``: geoVI via NIFTy (alias of ``vi``)
            - ``"vi_linear"``: MGVI via NIFTy (linearized Gaussian)
            - ``"vi_nonlinear_fast"``: geoVI fast path (~35% faster, no logging)
            - ``"vi_linear_fast"``: MGVI fast path (~35% faster, no logging)
            - ``"native_vi_nonlinear"``: Native JAX geoVI (**broken**: segfaults
              on DPL/dense_basis photometry mocks, issue #231)
            - ``"native_vi_linear"``: Native JAX MGVI (**broken**: same segfault)

            **MCMC Sampling**

            - ``"mcmc_nuts"``: NUTS via BlackJAX (default for D≤20; exact posterior)
            - ``"mcmc_raytrace"``: Ray Tracing (Behroozi 2025; O(1) gradient cost)
            - ``"mcmc"``: Auto: NUTS (D≤20) or Ray Tracing (D>20)
            - ``"mcmc_hmc"``: Standard HMC (fixed trajectory length)
            - ``"mcmc_dynamic_hmc"``: Dynamic HMC (adaptive trajectory)
            - ``"mcmc_ghmc"``: Generalized HMC (**broken**: R-hat ~ 2.5-3.1,
              ESS ~ 1 on D=6-7 mocks)
            - ``"mcmc_mclmc"``: MCLMC (**broken**: R-hat ~ 1.7, ESS ~ 1)
            - ``"mcmc_adjusted_mclmc"``: MCLMC + Metropolis correction
            - ``"mcmc_ess"``: Elliptical Slice Sampling (gradient-free)

            **Point Estimation & Approximations**

            - ``"map"``: MAP optimization (Adam by default)
            - ``"laplace"``: Laplace approximation (Gaussian posterior at MAP)
            - ``"pathfinder"``: L-BFGS trajectory + sequence of Gaussians (Zhang+2022)

            **Model Comparison (Bayesian Evidence)**

            - ``"nss"``: Nested Slice Sampling (exact Z, D≤30)

            **Automatic Selection**

            - ``"auto"``: NUTS (D≤20) or geoVI (D>20) based on dimensionality

        init_from : Posterior, optional
            Previous inference result to use as warm-start initialization.
            The posterior mean is extracted and converted to unbounded space.
            Useful for refining results across different methods. Default ``None``.

        key : PRNGKey, optional
            JAX random key. Default ``PRNGKey(42)`` for reproducibility.
            Ignored for deterministic methods (``"map"``, ``"laplace"``).

        allow_unvalidated : bool, optional
            Run a backend registered at ``tier="broken"`` — one that reports
            wrong answers or crashes in its own registry entry. Default
            ``False``, which raises :class:`~tengri.BackendError` naming the
            specific failure. Intended for benchmarking and backend
            development, not for science (#1287).

        prewarm : bool, optional
            JIT-compile the loss/gradient and the predict surface
            (``predict_photometry`` / ``predict_properties``) before the fit
            loop, populating the persistent cache so the fit runs warm and
            post-fit posterior-predictive / derived-quantity exploration is
            already compiled. Default ``True``; pass ``False`` for the previous
            lazy compile-on-first-call behavior. (The fit-time approximation
            policy is set with ``approx=`` on the :class:`Fitter` constructor /
            :meth:`ForwardModel.fit`, not here.)

        **kwargs
            Method-specific keyword arguments passed to the underlying backend:

            - **VI methods**: ``n_samples``, ``n_kl_iter``, ``tol_kl``, ``sample_mode``,
              ``verbose``, ``mirror_samples``.
            - **MCMC methods**: ``n_steps``, ``n_warmup``, ``thin``, ``step_size``,
              ``mass_matrix``, ``adapt_step_size``, ``verbose``.
            - **MAP/Laplace**: ``n_steps``, ``step_size``, ``lr``, ``verbose``.
            - **Pathfinder**: ``n_steps``, ``n_init``, ``step_size``, ``verbose``.
            - **NSS**: ``n_live``, ``n_batch``, ``slice_width``, ``verbose``.

            See backend docstrings for full option documentation.

        Returns
        -------
        Posterior
            Inference results object with attributes:

            - ``samples`` : dict or None — Posterior samples (None for MAP).
            - ``params`` : dict — Best-fit or posterior mean parameters.
            - ``method`` : str — Method used.
            - ``diagnostics`` : dict — Convergence/quality metrics.
            - ``log_evidence`` : float or None — Bayesian evidence (NSS only).
            - ``wall_time_s`` : float — Total runtime.

            The Posterior also has derived quantity methods:
            ``derived``, ``summary()``, ``to_arviz()``, ``refine()``, etc.

        Raises
        ------
        ParameterError
            If ``method`` is invalid or unrecognized.
        RuntimeError
            If background JIT compilation failed.
        ValueError
            If method-specific kwargs are invalid.

        Notes
        -----
        **Method selection strategy:**

        - **Default** (``"vi"``): geoVI is recommended for high-dimensional problems
          (D>50) and population fitting. Captures non-Gaussian posterior geometry.
        - **Exact posterior** (``"mcmc_nuts"``): Use for D≤20 where exact sampling is
          feasible and posterior validation is critical.
        - **Fast large-D sampling** (``"mcmc_raytrace"``): Use for D>50 with gradient
          access; 250× more robust to noisy gradients than HMC.
        - **Bayesian model comparison** (``"nss"``): Estimates log-evidence for
          comparing competing physical models (e.g., different dust laws).

        **Important gotchas:**

        - **VI posterior equivalence**: ``"vi"`` (NIFTy geoVI) and ``"vi_native"``
          (pure JAX geoVI) target the same objective but are NOT posterior-equivalent.
          The native version is ~19× faster but produces different posterior shapes
          on some problems (e.g., PSD timescale can differ by order of magnitude).
          Validate before swapping methods. See
          ``bench/reports/2026-04-17_native_vs_nifty.md`` in the repository
          (outside the docs tree, so it is not a linkable document).

        - **VIConfig.n_samples doubling**: In geoVI, when ``mirror_samples=True``
          (default), ``n_samples=3`` produces 6 effective samples (3 + 3 mirrors).
          When tuning convergence, think in effective samples.

        - **Ray Tracing step_size scaling**: Ray Tracing uses ``step_size=0.05`` by
          default for D~137. There is a sharp viability cliff at ~0.06 where
          acceptance drops from 80% to 0%. Use smaller step sizes for safety.

        - **Method defaults from file**: Default hyperparameters (``n_kl_iter``,
          ``n_warmup``, etc.) are loaded from ``defaults.toml`` if available.
          Command-line kwargs override file defaults.

        **Warm-starting from MAP:**

        Fitting often proceeds in stages:

        >>> result_map = fitter.run("map", n_steps=1500)
        >>> result_mcmc = fitter.run("mcmc_nuts", init_from=result_map, n_warmup=500)
        >>> result_vi = fitter.run("vi", init_from=result_map, n_samples=100)

        MAP provides a quick point estimate; MCMC and VI refine from this
        initialization, converging faster than from random initialization.

        **Reproducibility:**

        Pass ``key=jax.random.PRNGKey(seed)`` to control randomness across runs.
        ``key=None`` defaults to ``PRNGKey(42)`` for reproducibility.

        **Compile-cache behavior (smart lean, 2026-05):**

        ``run`` accepts a ``lean`` kwarg (default inferred from
        ``tengri.lean()`` / ``tengri.persistent()`` context). With
        ``lean=True`` (the default), the inference-body cache is
        cleared of *stale* entries — every entry whose
        ``(compile_signature, method)`` differs from the current call
        is dropped, but the entry that matches the current call (if it
        exists from a prior identical run) is kept. Forward-model,
        log-density, loss, and gradient compiles survive unconditionally.
        Implications:

        - Multi-phase notebooks (MAP → HMC → posterior-predictive)
          peak at one inference scan body in RAM, not several.
        - Catalog loops calling ``fitter.run(method)`` repeatedly with
          the same model and method pay one compile, not N — without
          needing ``tengri.persistent()``.

        ``tengri.gc()`` drops everything including structural caches;
        use it between loops that build many *different* model
        configurations.

        References
        ----------
        .. [1] M. D. Hoffman and A. Gelman, "The No-U-Turn Sampler: Adaptively
           Setting Path Lengths in Hamiltonian Monte Carlo," JMLR, 15, 1593 (2014).
           https://arxiv.org/abs/1111.4246

        .. [2] P. Behroozi, "The Ray Tracing Sampler," arXiv:2504.20029 (2025).
           https://arxiv.org/abs/2504.20029

        .. [3] L. Zhang et al., "Pathfinder: Parallel quasi-Newton variational
           inference," JMLR, 23, 306 (2022).
           https://arxiv.org/abs/2108.03782

        .. [4] B. D. Johnson et al., "Prospector: Stellar Population Inference
           from Spectra and SEDs," ApJS, 254, 22 (2021).
           arXiv:2012.01426. https://doi.org/10.3847/1538-4365/abef67

        Examples
        --------
        **Example 1: Quick exploration with MAP + geoVI**

        >>> fitter = Fitter(forward, data, noise)
        >>> result = fitter.run("vi")  # geoVI with defaults
        >>> print(result.summary())

        **Example 2: Exact posterior with NUTS (small-D)**

        >>> result = fitter.run("mcmc_nuts", n_warmup=500, n_steps=2000)
        >>> samples = result.samples["stellar_mass"]
        >>> print(f"M_star = {jnp.median(samples):.2e} Msun")

        **Example 3: Warm-start MCMC from MAP**

        >>> result_map = fitter.run("map", n_steps=1500)
        >>> result_mcmc = fitter.run("mcmc_nuts", init_from=result_map, n_warmup=300, n_steps=1000)

        **Example 4: Nested sampling for Bayesian model comparison**

        >>> result_nss = fitter.run("nss", n_live=100)
        >>> log_z = result_nss.log_evidence
        >>> print(f"log(Z) = {log_z:.2f}")  # Use for Bayes factors

        **Example 5: Using ``"auto"`` method for unknown dimensionality**

        >>> result = fitter.run("auto")  # NUTS if D≤20, VI if D>20
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        # Resolve deprecated aliases and validate method
        method = resolve_method(method)

        # #1671 made operational: this fit runs on a resolved precompute LUT,
        # so price the LUT's forward bias against this fit's SNR once — the
        # amplified estimate (bias x SNR) is what moves the posterior mode,
        # and no forward-model check can see it. Once per fitter instance.
        if not self._lut_bias_checked and self._pre_approx_model is not None:
            self._lut_bias_checked = True
            _warn_if_lut_bias_amplified(
                self._pre_approx_model,
                self.model,
                self.data,
                self.noise,
                self.data_type,
                surface="Fitter",
            )

        # --- Smart lean: drop only stale L3 entries before this run ---
        # Default: lean=True. The smart-lean path (below) preserves the
        # entry whose key matches this fitter's compile_signature, so
        # CatalogFitter loops and repeated identical fits hit the cache
        # without any opt-in. ``tengri.persistent()`` is rarely needed —
        # it only matters if you want to keep *non-matching* entries
        # alive (e.g. swapping back and forth between MAP and HMC and
        # wanting both compiles in RAM). Override per-call via
        # ``fitter.run(..., lean=True/False)``.
        import warnings

        from tengri.inference.jit_engine import (
            clear_shared_caches as _clear_shared_caches,
            is_lean_mode as _is_lean_mode,
            is_persistent_mode as _is_persistent_mode,
        )

        # --- Lean kwarg deprecation (issue #1318) ─────────────────────────
        # The lean= kwarg is retired. Callers that pass it get a one-shot
        # DeprecationWarning with the retire message. The behavior is still
        # honored for back-compat. For new code, the cache policy is derived
        # from a private _cache_policy kwarg (set by Catalog) or defaults to
        # "iterate" (keep-matching smart-lean behavior).
        _user_lean = kwargs.pop("lean", None)
        if _user_lean is not None:
            warnings.warn(
                "lean= is retired: every fit keeps its warm caches (iterate "
                "policy), including catalog fits — one inference-body compile "
                "serves the whole catalog. See #1318, #1344.",
                DeprecationWarning,
                stacklevel=2,
            )

        # --- Cache policy derivation (2026-07) ────────────────────────────
        # _cache_policy selects the L3 (inference-body) eviction policy.
        #
        # Catalog deliberately does NOT pass 'sweep' (#1344). ``_lean_keep_sig``
        # is ``compile_signature()`` — data *shape*, never data values — so two
        # galaxies of the same model and shape produce the same key. 'iterate'
        # therefore keeps the one entry both galaxies share and the catalog pays
        # one inference-body compile; 'sweep' passes ``keep_sig=None`` and drops
        # that entry too, which would recompile per galaxy — the #1316 cliff the
        # catalog path exists to avoid. 'sweep' remains reachable through
        # ``tengri.lean()`` for memory-constrained runs that would rather pay
        # the recompile. If not present,
        # derive based on the deprecated lean= kwarg (for back-compat) or the
        # context (persistent() vs lean()). Default: 'iterate' (keep-matching).
        _cache_policy = kwargs.pop("_cache_policy", None)
        if _cache_policy is None:
            # Back-compat: if lean= was passed, honor it
            if _user_lean is not None:
                _cache_policy = "sweep" if _user_lean else "iterate"
            else:
                # Derive from context (persistent/lean/default)
                if _is_persistent_mode():
                    # persistent() promises to keep ALL cached artifacts, including
                    # non-matching L3 entries — a distinct policy, NOT "iterate"
                    # (which drops stale non-matching entries). Mapping it to
                    # "iterate" made persistent() a silent no-op.
                    _cache_policy = "persistent"
                elif _is_lean_mode():
                    _cache_policy = "sweep"  # Drop all stale entries
                else:
                    _cache_policy = "iterate"  # Default: keep-matching

        # Apply cache policy. The policy semantic:
        # - "iterate": drop only L3 entries that do NOT match this fitter's
        #   compile_signature() (smart-lean behavior, 2026-05)
        # - "sweep": drop all L3 entries (old lean=True behavior)
        # The compile_signature() keys on shape, not values — so identical
        # geometry always matches, even if parameters differ.
        if _cache_policy == "sweep":
            _clear_shared_caches(scope="inference_body", keep_sig=None)
        elif _cache_policy == "iterate":
            # Keep-matching: preserve the entry matching this fitter's signature
            # Smart lean (2026-05): drop only L3 entries that do NOT match
            # this fitter's compile_signature(). The matching entry — if
            # it exists from a prior identical run — is kept, so a
            # CatalogFitter loop or repeated identical fitter.run() call
            # hits the cache instead of recompiling. The engine cache
            # (``_SHARED_ENGINE_CACHE``) is keyed on the bare
            # ``compile_signature()``; the engine itself contains
            # compiled functions for every method, so per-method
            # invalidation is unnecessary. Forward, loss, grad, and
            # logdensity caches are preserved unconditionally at this
            # scope.
            # drop_xla=False (#1350): this policy's promise is "fit() keeps your
            # warm caches". jax.clear_caches() would wipe the process-wide XLA
            # executables, leaving the entries we deliberately keep as hollow
            # shells that re-trace — and de-warming the caller's own
            # predict_photometry too. The stale non-matching tengri entries are
            # still dropped. "sweep" keeps drop_xla=True: it exists for memory
            # relief (the notebook-OOM class) and must keep releasing executables.
            _clear_shared_caches(
                scope="inference_body", keep_sig=self._lean_keep_sig, drop_xla=False
            )
        elif _cache_policy == "persistent":
            # persistent(): keep EVERYTHING — no L3 clear at all, even
            # non-matching stale entries. This is what the context manager /
            # TENGRI_PERSISTENT promise, and it is distinct from "iterate".
            pass

        # --- Merge TOML method-specific defaults (caller kwargs win) ---
        try:
            from tengri.parameters.defaults import get_inference_defaults

            kwargs = {**get_inference_defaults(method), **kwargs}
        except (ImportError, FileNotFoundError, OSError):
            # Config file unavailable or unreadable — skip defaults merge
            pass

        # Strip any stale vi_flavor kwarg that callers may pass (no longer used)
        kwargs.pop("vi_flavor", None)

        # Extract memory/chunking controls. They are orthogonal to inference
        # method, so we pluck them from kwargs here and stash on the Fitter
        # rather than plumbing them through every _run_* backend signature.
        # - memory_mode="auto" picks "low" for stochastic (high-D field)
        #   models, "fast" otherwise. "low" wraps signal_response in
        #   jax.checkpoint — 2-3x peak memory reduction inside CG at a
        #   modest wall-time cost. Used via _engine_cache_key so different
        #   modes get separate cached engines.
        # - posterior_chunk_size controls peak memory of _draw_jit_samples
        #   (see _draw_posterior_samples docstring).
        memory_mode = kwargs.pop("memory_mode", "auto")
        if memory_mode == "auto":
            memory_mode = "low" if self.spec.stochastic else "fast"
        if memory_mode not in ("fast", "low"):
            raise ValueError(f"memory_mode must be 'auto', 'fast', or 'low' (got {memory_mode!r})")
        if getattr(self, "_memory_mode", None) != memory_mode:
            # Invalidate the per-instance engine reference — a different
            # memory_mode needs a different cached engine.
            self._jit_sampler = None
        self._memory_mode = memory_mode
        self._posterior_chunk_size = kwargs.pop("posterior_chunk_size", None)
        # Auto-prewarm the JIT (loss/grad + sampler + predict surface) before the
        # fit loop unless opted out. Default True; internal throwaway runs from
        # prewarm() pass prewarm=False to avoid re-entry.
        prewarm = kwargs.pop("prewarm", True)

        # --- "auto" method: dimensionality-based selection ---
        if method == "auto":
            d = self.spec.n_latent
            try:
                from tengri.parameters.defaults import get_inference_defaults

                threshold = int(get_inference_defaults().get("mcmc_auto_d", _AUTO_D_THRESHOLD))
            except (ImportError, FileNotFoundError, OSError, KeyError, ValueError):
                # Config unavailable, missing key, or non-integer value — use hardcoded fallback
                threshold = _AUTO_D_THRESHOLD
            method = "mcmc_nuts" if d <= threshold else "vi_nonlinear_fast"

        # --- Dispatch to underlying _run_* methods via registry ---
        from tengri.inference._backend_registry import (
            check_capabilities,
            check_requires,
            check_unknown_kwargs,
            check_usable,
            get_backend,
        )

        if method == "auto":
            # Pre-registry semantics: low-D → NUTS (exact), high-D → geoVI (scalable).
            # Dimensionality threshold is configurable via inference defaults.
            d = self.spec.n_latent
            try:
                from tengri.parameters.defaults import get_inference_defaults

                threshold = int(
                    get_inference_defaults().get("mcmc_auto_d", _MCMC_AUTO_D_THRESHOLD)
                )
            except (ImportError, FileNotFoundError, OSError, KeyError, ValueError):
                threshold = _MCMC_AUTO_D_THRESHOLD
            chosen = "mcmc_nuts" if d <= threshold else "vi_nonlinear_fast"
            print(
                f"  [auto-pick]  D={d} (threshold {threshold}) → "
                f"using '{chosen}'.  Pass an explicit method= to override."
            )
            entry = get_backend(chosen)
        elif method == "mcmc":
            # Auto-select within the MCMC family: NUTS for low-D, ray tracing for high-D.
            d = self.spec.n_latent
            if d <= _MCMC_AUTO_D_THRESHOLD:
                chosen = "mcmc_nuts"
            else:
                chosen = "mcmc_raytrace"
            print(
                f"  [mcmc auto-pick]  D={d} (threshold {_MCMC_AUTO_D_THRESHOLD}) → "
                f"using '{chosen}'.  Pass method='mcmc_nuts' or 'mcmc_raytrace' to override."
            )
            entry = get_backend(chosen)
        else:
            entry = get_backend(method)

        # Pre-flight speed guard: steer many-evaluation samplers off the slow
        # exact forward path onto the WavePrecomp LUT (see helper above).
        _warn_if_exact_forward_path(self.model, entry.name)

        # Pre-flight memory guard. `auto`/`mcmc` already switch away from NUTS
        # above D=20, but an explicit method='mcmc_nuts' overrides nothing — the
        # caller has chosen, so tell them the cost instead of silently paying it.
        # Same helper serves CatalogFitter and PopulationFitter (#1394 follow-up).
        _warn_if_nuts_high_dim(entry.name, self.spec.n_free, surface="Fitter.run")

        # Refuse backends that declare themselves unusable, unless the caller
        # opts in explicitly (#1287). Before check_requires, because "this
        # sampler returns R-hat ~ 3" is a more fundamental objection than
        # "its optional dependency is missing".
        check_usable(entry, allow_unvalidated=allow_unvalidated)

        # Friendly error if the backend's optional dependency is missing,
        # before we descend into a deep third-party traceback.
        check_requires(entry)

        # Refuse capability kwargs this backend cannot honor. Without it,
        # ``precondition=True`` on a non-Hamiltonian method travels until a terminal
        # function without ``**kwargs`` rejects it, surfacing as a TypeError naming
        # ``run_nifty_vi`` or ``run_map`` — functions the caller never mentioned.
        # Raises ValueError, matching the answer ``method='mcmc'`` already gave.
        check_capabilities(entry, kwargs)
        # Same answer for every other unrecognized name, so a typo or an
        # unsupported channel names the method instead of the runner (#1469).
        # Constructor-routed names go in as suggestion candidates only: they
        # are documented fit() options, so a typo'd one must be correctable
        # even though the runner does not declare it.
        check_unknown_kwargs(entry, kwargs, also_accepted=fit_surface_ctor_names())

        # Compile the loss/grad + predict surface up front so the fit loop runs
        # warm and the persistent JAX cache is populated.
        if prewarm:
            self._auto_prewarm(key)

        # Build the Python-level InferenceContext once. Backends marked
        # ``legacy_fitter=True`` continue to receive the full Fitter
        # (their lambdas at the bottom of this file relay to ``_run_*``
        # methods); migrated backends receive the context and access
        # state through its explicit accessors. See ADR-0010 / context.py.
        from tengri.inference.context import InferenceContext

        context = InferenceContext(fitter=self)
        target = self if entry.legacy_fitter else context
        result = entry.runner(target, key=key, init_from=init_from, **kwargs)

        # Attach back-reference so Posterior.refine() works
        with contextlib.suppress(AttributeError):
            result._fitter = self
        # Record the canonical registry key for citation collection.
        # ``Posterior.method`` is a display string ("NUTS (BlackJAX)"), not a
        # key, so the sampler citation cannot be recovered from it later.
        with contextlib.suppress(AttributeError):
            result._backend_key = entry.name
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the fitting problem.

        Returns
        -------
        str
            Formatted summary showing data shape, free parameters,
            priors, bounds, and available inference methods.

        Notes
        -----
        The summary includes:

        - Data dimensionality and median signal-to-noise ratio
        - Free parameters and latent grid points (ξ) if stochastic SFH
        - Parameter names, prior distributions, and bounds
        - All available inference methods

        Examples
        --------
        >>> fitter = Fitter(forward, data, noise)
        >>> print(fitter.summary())
        Fitter  data_type: photometry
        ──────────────────────────────────────────────────────────────
          Data points: 100
          Median S/N:  5.2
          Parameters:  8 free + 64 latent (ξ)
        ...
        """
        sep = "─" * 66
        lines: list[str] = [f"Fitter  data_type: {self.data_type}", sep]

        # Data shape
        n_data = self.data.shape[0]
        snr_med = float(jnp.median(jnp.abs(self.data / self.noise)))
        lines.append(f"  Data points: {n_data}")
        lines.append(f"  Median S/N:  {snr_med:.1f}")

        # Dimensionality
        n_free = len(self._free_names)
        n_grid = self.model.n_grid if self.model.uses_stochastic_sfh else 0
        dim_str = f"{n_free} free"
        if n_grid:
            dim_str += f" + {n_grid} latent (ξ)"
        lines.append(f"  Parameters:  {dim_str}")
        lines.append("")

        # Free parameter table
        hdr = f"  {'Parameter':<32s} {'Prior':<26s} {'Bounds'}"
        lines.append(hdr)
        lines.append("  " + "─" * 64)
        for name in self._free_names:
            dist = self.spec.get_distribution(name)
            lo, hi = dist.bounds
            lines.append(f"  {name:<32s} {dist!r:<26s} [{lo:.4g}, {hi:.4g}]")

        # Available methods
        lines.append("")
        lines.append(
            "  Methods:     vi, vi_linear, vi_nonlinear_fast, vi_linear_fast, "
            "vi_native, vi_native_linear, mcmc, mcmc_raytrace, mcmc_nuts, "
            "mcmc_hmc, mcmc_dynamic_hmc, mcmc_ghmc, mcmc_mclmc, "
            "mcmc_adjusted_mclmc, mcmc_ess, map, laplace, pathfinder, nss, auto"
        )

        lines.append(sep)
        return "\n".join(lines)

    # ── Private method runners ────────────────────────────────────────
    #
    # Public dispatch does NOT pass through methods here: ``Fitter.run()``
    # resolves the method name in the backend registry
    # (``inference/_registration.py``) and calls the registered runner with
    # an ``InferenceContext`` (ADR-0010). The per-method ``_run_*`` shims
    # that used to mirror every backend were deleted 2026-07 once the
    # registry migration completed; ``_run_map`` alone survives because
    # warm-start paths (native VI, ``_sample_utils``) call it directly.

    def _run_map(self, *, key, **kwargs) -> Posterior:
        """Dispatch to MAP optimization via gradient descent (Adam by default)."""
        from tengri.inference.backends.map_dispatch import run_map

        return run_map(self, key=key, **kwargs)

    # ── Posterior sampling ────────────────────────────────────────────

    def _draw_posterior_samples(
        self,
        likelihood,
        pos_dict,
        key,
        n_samples,
        existing_samples,
        *,
        method="jit",
        posterior_chunk_size=None,
        verbose=True,
    ):
        """Draw posterior samples from the converged geoVI approximation.

        Parameters
        ----------
        method : str
            "jit" (default) — JIT-compiled CG solve, ~0.2ms/sample.
            "blackjax" — BlackJAX NUTS (independent MCMC, not geoVI).
            "nifty" — NIFTy draw_linear_residual (slow, ~540ms/sample).
        posterior_chunk_size : int, optional
            If set, process CG draws in chunks of this size — peak memory
            becomes O(chunk · D) instead of O(n_samples · D). JIT cache
            hits across chunks, so wall-time overhead is negligible.
        """
        if method == "jit":
            return self._draw_jit_samples(
                pos_dict,
                key,
                n_samples,
                existing_samples,
                posterior_chunk_size=posterior_chunk_size,
                verbose=verbose,
            )
        if method == "blackjax":
            try:
                return self._draw_blackjax_samples(
                    likelihood, pos_dict, key, n_samples, existing_samples, verbose=verbose
                )
            except ImportError:
                if verbose:
                    logger.info("  blackjax not installed, falling back to JIT sampling")
                return self._draw_jit_samples(
                    pos_dict,
                    key,
                    n_samples,
                    existing_samples,
                    posterior_chunk_size=posterior_chunk_size,
                    verbose=verbose,
                )
        return self._draw_nifty_samples(
            likelihood, pos_dict, key, n_samples, existing_samples, verbose=verbose
        )

    def _draw_jit_samples(
        self,
        pos_dict,
        key,
        n_samples,
        existing_samples,
        *,
        posterior_chunk_size=None,
        verbose=True,
    ):
        """Draw geoVI linear residual samples via JIT-compiled CG.

        Same math as NIFTy's draw_linear_residual but fully JIT-compiled:
        1. Draw z = J^T sqrt(N^{-1}) eta1 + eta2  (eta_i ~ N(0,I))
        2. Solve M @ residual = z via CG  (M = J^T N^{-1} J + I)
        3. Sample = pos + residual

        ~2000x faster than NIFTy's Python-loop CG.

        When ``posterior_chunk_size`` is set, the call to
        ``engine["draw_samples"]`` is split into fixed-size chunks so peak
        memory is O(chunk · D) instead of O(n_samples · D). Chunks are
        padded to stable size so the JIT cache hits across calls.
        """
        if verbose:
            logger.info("  Drawing %d posterior samples (JIT CG)...", n_samples)

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(pos_dict)

        engine = self._jit_sampler
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        pos_flat = flatten(pos_dict)
        draw_keys = jax.random.split(key, n_samples)
        data_args = self._data_args

        # Resolve the effective chunk size. Precedence:
        #   1. explicit kwarg (caller wins)
        #   2. stashed on self by Fitter.run(posterior_chunk_size=...)
        #   3. auto-chunk of 64 when memory_mode="low"
        #      (jax.checkpoint + jax.vmap(N_large) holds all N
        #      recomputed forwards simultaneously — negating most
        #      of the memory saving. Chunking keeps that to
        #      O(64 · activations) regardless of n_samples.)
        #   4. otherwise unchunked (preserves prior behavior)
        if posterior_chunk_size is None:
            posterior_chunk_size = getattr(self, "_posterior_chunk_size", None)
        if posterior_chunk_size is None and getattr(self, "_memory_mode", "fast") == "low":
            posterior_chunk_size = 64
        chunk = posterior_chunk_size if posterior_chunk_size else n_samples
        chunk = min(int(chunk), int(n_samples))
        if chunk >= n_samples:
            residuals_flat = engine["draw_samples"](pos_flat, draw_keys, data_args)
        else:
            parts = []
            for start in range(0, n_samples, chunk):
                end = min(start + chunk, n_samples)
                keys_chunk = draw_keys[start:end]
                pad = chunk - (end - start)
                if pad:
                    keys_chunk = jnp.concatenate([keys_chunk, draw_keys[:pad]])
                r = engine["draw_samples"](pos_flat, keys_chunk, data_args)
                jax.block_until_ready(r)
                if pad:
                    r = r[: end - start]
                parts.append(r)
            residuals_flat = jnp.concatenate(parts, axis=0)

        for i in range(n_samples):
            res = unflatten(residuals_flat[i])
            combined = {k: pos_dict[k] + res[k] for k in pos_dict}
            existing_samples.append(combined)

        return existing_samples

    def _draw_nonlinear_jit_samples(
        self,
        pos_dict,
        key,
        n_samples,
        existing_samples,
        *,
        posterior_chunk_size=None,
        verbose=True,
    ):
        """Draw geoVI nonlinear posterior samples via JIT engine.

        Unlike ``_draw_jit_samples`` (linear CG only), this applies
        the geoVI coordinate curving to each sample.  Produces
        samples from the nonlinear approximation, capturing
        banana-shaped degeneracies that the linear Gaussian misses.

        Uses ``draw_nonlinear_residuals`` from the JIT engine.
        """
        if verbose:
            logger.info("  Drawing %d nonlinear posterior samples (JIT geoVI)...", n_samples)

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(pos_dict)

        engine = self._jit_sampler
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        pos_flat = flatten(pos_dict)
        data_args = self._data_args

        # Draw in batches to avoid OOM for large n_samples. Default 50 is
        # the pre-existing safety cap; posterior_chunk_size overrides it.
        # With memory_mode="low" we tighten the default to 64 to match the
        # linear-draw path (checkpoint+vmap anti-pattern, see
        # _draw_jit_samples docstring).
        if posterior_chunk_size is None:
            posterior_chunk_size = getattr(self, "_posterior_chunk_size", None)
        if posterior_chunk_size is None and getattr(self, "_memory_mode", "fast") == "low":
            posterior_chunk_size = 64
        batch_size = int(posterior_chunk_size) if posterior_chunk_size else 50
        batch_size = min(n_samples, batch_size)
        draw_keys = jax.random.split(key, n_samples)

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_keys = draw_keys[batch_start:batch_end]
            # draw_nonlinear_samples returns (2*n, D): first n positive, last n mirrors
            residuals_flat = engine["draw_nonlinear_samples"](pos_flat, batch_keys, data_args)
            n_batch = batch_end - batch_start
            # Use only the first n (positive) samples, not the mirrors
            for i in range(n_batch):
                res = unflatten(residuals_flat[i])
                combined = {k: pos_dict[k] + res[k] for k in pos_dict}
                existing_samples.append(combined)

        return existing_samples

    def _draw_blackjax_samples(
        self, likelihood, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw samples via BlackJAX NUTS (independent MCMC, not geoVI)."""
        from tengri.inference.backends.mcmc._shared import _check_blackjax_floor

        _check_blackjax_floor()
        import blackjax

        if verbose:
            logger.info("  Drawing %d posterior samples via BlackJAX NUTS...", n_samples)

        warmup_key, sample_key = jax.random.split(key)
        n_warmup = min(200, n_samples)

        # Warmup and sampling run inside ONE memoized ``jax.jit`` that takes the data as
        # a *traced* argument. The eager form rebuilt a fresh ``@jax.jit logdensity_fn``
        # (closing over ``self._data_args``, which carries the SSP grid), a fresh
        # ``window_adaptation``, a fresh NUTS kernel and a fresh ``one_step`` on every
        # call. Fresh function identities miss JAX's compilation cache, so each call
        # compiled and *retained* another set of executables — a measured ~72 MB/call
        # leak (#1249), accrued per galaxy by a catalog VI run drawing samples with
        # ``posterior_method="blackjax"``.
        #
        # Memoizing only the log-density is not enough (measured: 72 -> 55 MB/call);
        # the adaptation-dependent kernel has to be inside the cached program too, and
        # it cannot be cached separately because it is built from the *runtime* warmup
        # output — caching that would bake one call's step size, the bug fixed in #1234.
        def _build_draw(ld_from):
            """Compile (window adaptation -> sampling scan) once, data traced."""

            def _run(wk, sk, pos, data_args):
                """Window-adapt, then scan the NUTS kernel; returns stacked positions."""
                ld = ld_from(data_args)
                warmup = blackjax.window_adaptation(blackjax.nuts, ld)
                (state, parameters), _ = warmup.run(wk, pos, num_steps=n_warmup)
                kernel = blackjax.nuts(ld, **parameters).step

                def one_step(s, rng_key):
                    """One NUTS step, shaped for ``jax.lax.scan``."""
                    s, _ = kernel(rng_key, s)
                    return s, s

                _, states = jax.lax.scan(one_step, state, jax.random.split(sk, n_samples))
                return states.position

            return jax.jit(_run)

        if likelihood is not None:
            # A caller-supplied likelihood is an arbitrary callable we cannot
            # fingerprint, so the memo is disabled (``key=None`` builds fresh) rather
            # than risk serving a kernel compiled around a different likelihood.
            def _ld_from(_data_args):
                """Custom-likelihood log posterior (likelihood + standard normal prior)."""

                def _ld(x):
                    prior = 0.5 * sum(jnp.sum(v**2) for v in x.values())
                    return -likelihood(x) - prior

                return _ld

            memo_key = None
        else:
            _logdensity_2arg = self._get_or_build_logdensity_fn()

            def _ld_from(data_args):
                """Default log posterior, reading the traced ``data_args``."""

                def _ld(x):
                    return _logdensity_2arg(x, data_args)

                return _ld

            memo_key = (
                "blackjax_draw",
                self.compile_signature(),
                int(n_samples),
                int(n_warmup),
            )

        _run_draw = self._memo_batch_kernel(
            "_blackjax_draw_kernel_cache", memo_key, lambda: _build_draw(_ld_from)
        )
        sample_positions = _run_draw(warmup_key, sample_key, pos_dict, self._data_args)

        if verbose:
            logger.info("  Warmup done (%d steps). Sampled %d draws.", n_warmup, n_samples)

        for i in range(n_samples):
            sd = jax.tree.map(lambda x, _i=i: x[_i], sample_positions)
            existing_samples.append(sd)

        return existing_samples

    def _draw_nifty_samples(
        self, likelihood, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw samples via NIFTy's draw_linear_residual (slow, ~540ms/sample)."""
        import nifty8.re as jft

        if verbose:
            logger.info("  Drawing %d posterior samples (NIFTy CG)...", n_samples)

        converged_pos = jft.Vector(pos_dict)
        draw_keys = jax.random.split(key, n_samples)
        for sub_key in draw_keys:
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood,
                    converged_pos,
                    sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 30},
                )
                sample_tree = residual.tree if hasattr(residual, "tree") else dict(residual)
                pos_tree = (
                    converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
                )
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                existing_samples.append(combined)
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
                # TypeError: NIFTy API mismatch or dict() conversion failed
                # ValueError: invalid cg_kwargs configuration
                # AttributeError: missing .tree attribute
                # KeyError: position/sample tree key mismatch
                # RuntimeError: linear solver failed to converge
                # Stop generating warmup samples and return what we have
                break

        return existing_samples

    # ── Batch ─────────────────────────────────────────────────────────

    def fit_batch(
        self,
        batch,
        *,
        method="vi",
        key=None,
        verbose=True,
        **kwargs,
    ):
        """Fit a batch of galaxies efficiently.

        Creates a Fitter per galaxy, sharing the XLA compilation cache.
        The first galaxy pays compile cost; subsequent galaxies load
        from the persistent XLA cache (milliseconds each).

        Works with any inference method — vi (default) gives
        the best speed. Also usable for hierarchical individual fits.

        Parameters
        ----------
        batch : list of dict
            Each dict has "flux_obs" and "noise" arrays.
        method : str
            Default "vi". Any method from run().
        key : PRNGKey, optional
            Random seed for sampling methods. Default: ``jax.random.PRNGKey(42)``.
        verbose : bool
            Print progress. Default: ``True``.
        **kwargs
            Passed to run() (n_iterations, n_samples, n_seeds, etc).

        Returns
        -------
        list of Posterior
            Inference results for each galaxy, in order.

        Notes
        -----
        **Parallelization strategy**:

        - For ``method="map"`` with precomputed photometry: uses ``jax.vmap``
          to fit all galaxies in a single JIT call (1-2s total).
        - For MCMC methods with fixed SFH: uses ``jax.vmap`` + shared adaptation.
        - Otherwise: sequential Fitter per galaxy (load from XLA cache).

        **Compilation caching**: All Fitters share the same Model instance,
        enabling persistent XLA cache. After first galaxy, subsequent fits
        are 10-100× faster depending on method.

        **Native VI tuning**: When ``method`` contains ``"native"`` and
        ``n_seeds`` is not explicitly passed, automatically sets ``n_seeds=5``
        for better convergence.

        Examples
        --------
        Batch fit 100 galaxies:

        >>> batch = [{"flux_obs": f, "noise": n} for f, n in zip(fluxes, noises)]
        >>> results = fitter.fit_batch(batch, method="vi")
        >>> # First: ~2s compile. Rest: ~2ms each. Total: ~0.2s per galaxy.

        Warm-start from MAP:

        >>> results_map = fitter.fit_batch(batch, method="map", n_steps=500)
        >>> results_vi = fitter.fit_batch(batch, method="vi", init_from=results_map)
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        if "native" in method and "n_seeds" not in kwargs:
            kwargs["n_seeds"] = 5

        n_gal = len(batch)
        if verbose:
            logger.info("fit_batch: %d galaxies, method=%s", n_gal, method)

        # vmap batch MAP: vectorize optimization over all galaxies in one JIT call.
        # Enabled when: method="map", precomp is set (same model for all galaxies),
        # and all galaxies have the same data shape.
        _same_shape = n_gal > 1 and all(
            jnp.asarray(g["flux_obs"]).shape == jnp.asarray(batch[0]["flux_obs"]).shape
            for g in batch
        )
        _use_vmap_map = (
            method == "map" and self.model.has_fixedz_photometry_precompute and _same_shape
        )

        if _use_vmap_map:
            return self._fit_batch_vmap_map(batch, key=key, verbose=verbose, **kwargs)

        # vmap batch MCMC: vectorize sampling over all galaxies in one JIT call.
        _mcmc_methods = {
            "mcmc_nuts",
            "mcmc_hmc",
            "mcmc_dynamic_hmc",
            "mcmc_ghmc",
        }
        _use_vmap_mcmc = (
            method in _mcmc_methods
            and self.model.has_fixedz_photometry_precompute
            and _same_shape
            and not self.spec.stochastic
        )

        if _use_vmap_mcmc:
            return self._fit_batch_vmap_mcmc(
                batch,
                key=key,
                method=method,
                verbose=verbose,
                **kwargs,
            )

        results = []
        t0 = time.time()

        for i, gal in enumerate(batch):
            gal_key = jax.random.fold_in(key, i)
            t_gal = time.time()

            fitter_i = Fitter(
                self.model,
                gal["flux_obs"],
                gal["noise"],
                data_type=self.data_type,
            )
            result_i = fitter_i.run(method, key=gal_key, verbose=False, **kwargs)
            results.append(result_i)

            dt = time.time() - t_gal
            if verbose and (i < 3 or (i + 1) % max(1, n_gal // 10) == 0 or i == n_gal - 1):
                chi2 = result_i.diagnostics.get("chi2_dof", "?")
                chi2_str = f"{chi2:.2f}" if isinstance(chi2, float) else str(chi2)
                logger.info("  Galaxy %d/%d: chi2/dof=%s, %.1fs", i + 1, n_gal, chi2_str, dt)

        t_total = time.time() - t0
        if verbose:
            logger.info(
                "  Done: %d galaxies in %.1fs (%.1fs/galaxy)",
                n_gal,
                t_total,
                t_total / n_gal,
            )

        return results

    def _fit_batch_vmap_mcmc(
        self,
        batch,
        *,
        key,
        method="mcmc_nuts",
        verbose=True,
        **kwargs,
    ):
        """Vectorized MCMC sampling over a batch using jax.vmap.

        All galaxies share the same compiled XLA kernel — adaptation
        parameters are computed on the first galaxy and reused for all.
        A single ``jax.jit(jax.vmap(...))`` call runs sampling for all
        galaxies in parallel.

        Requirements: same model structure, same data shape, parametric SFH.
        """
        from jax.flatten_util import ravel_pytree

        from tengri.inference.backends.mcmc._shared import (
            DEFAULT_MAX_NUM_DOUBLINGS,
            _check_blackjax_floor,
            _get_dynamic_hmc_kernel,
            _get_flat_logdensity,
            _get_ghmc_kernel,
            _get_hmc_kernel,
            _get_nuts_kernel,
        )

        _check_blackjax_floor()
        import blackjax

        from tengri.inference.posterior import Posterior

        n_warmup = kwargs.get("n_warmup", 300)
        n_burnin = kwargs.get("n_burnin", 100)
        n_samples = kwargs.get("n_samples", 1000)
        target_accept_rate = kwargs.get("target_accept_rate", 0.85)
        max_num_doublings = kwargs.get("max_num_doublings", DEFAULT_MAX_NUM_DOUBLINGS)
        dense_mass_matrix = kwargs.get("dense_mass_matrix", True)
        n_leapfrog_steps = kwargs.get("n_leapfrog_steps", 10)
        alpha = kwargs.get("alpha", 0.8)
        delta = kwargs.get("delta", 0.1)

        n_gal = len(batch)
        t0 = time.time()

        # All galaxies must have the same band count — vmap requires uniform shapes.
        n_obs_set = {len(g["flux_obs"]) for g in batch}
        if len(n_obs_set) != 1:
            raise ValueError(
                f"_fit_batch_vmap_mcmc requires all galaxies to have the same number of "
                f"observations, but got sizes: {sorted(n_obs_set)}. "
                "Use fit_batch with vmap=False to handle heterogeneous data."
            )

        # Stack galaxy data into batch arrays (n_gal, n_obs)
        flux_batch = jnp.stack([jnp.asarray(g["flux_obs"]) for g in batch])
        noise_batch = jnp.stack([jnp.asarray(g["noise"]) for g in batch])
        batch_data_args = {
            "data": flux_batch,
            "noise": noise_batch,
            "sqrt_noise_inv": inv_noise_std(noise_batch),
        }

        # Get shared logdensity (cached on Model, stable identity)
        init_params = self._initialize_unbounded(jax.random.PRNGKey(0))
        logdensity_flat_2arg, unravel_fn, _, _ = _get_flat_logdensity(self, init_params)

        # Initialize params per galaxy
        init_keys = jax.random.split(key, n_gal + 2)
        key = init_keys[0]
        adapt_key = init_keys[1]
        init_params_list = [self._initialize_unbounded(init_keys[2 + i]) for i in range(n_gal)]
        init_flats = jnp.stack([ravel_pytree(p)[0] for p in init_params_list])

        n_dim = init_flats.shape[1]
        use_dense = dense_mass_matrix and n_dim <= 30

        # Adaptation on the first galaxy, shared across the batch. Wrapped in a
        # memoized jax.jit that takes the galaxy data as a *traced* argument so the
        # compiled warmup is reused across fit_batch calls. The eager form built a
        # fresh ``window_adaptation`` (with a fresh log-density closure) every call
        # — a fresh function identity that JAX's in-memory compile cache never
        # reused, so it retained one warmup executable per call (~20 MB/call leak on
        # a long-lived Fitter). Data enters traced, so the adaptation still runs on
        # the current galaxy each call and the numbers are unchanged; this mirrors
        # the single-galaxy path's module-level ``_hmc_full_scan`` jit.
        first_data_args = jax.tree.map(lambda x: x[0], batch_data_args)

        if method == "mcmc_hmc":
            _adapt_algo, _adapt_kwargs = blackjax.hmc, {"num_integration_steps": n_leapfrog_steps}
        elif method == "mcmc_dynamic_hmc":
            _adapt_algo, _adapt_kwargs = blackjax.hmc, {"num_integration_steps": 10}
        else:  # mcmc_nuts, mcmc_ghmc — both tune with the NUTS window
            _adapt_algo, _adapt_kwargs = blackjax.nuts, {}

        def _run_window_adaptation(adapt_key, init_flat, data_args_first):
            """Window-adapt step size / mass matrix on the first galaxy.

            ``data_args_first`` is a *traced* argument, so this compiles once and
            is reused across ``fit_batch`` calls (memoized below). Not public API.
            """

            def _ld(pos):
                """Single-galaxy log-density closing over the traced data."""
                return logdensity_flat_2arg(pos, data_args_first)

            warmup = blackjax.window_adaptation(
                _adapt_algo,
                _ld,
                is_mass_matrix_diagonal=not use_dense,
                target_acceptance_rate=target_accept_rate,
                **_adapt_kwargs,
            )
            (_, adapt_params), _ = warmup.run(adapt_key, init_flat, num_steps=n_warmup)
            return adapt_params["step_size"], adapt_params["inverse_mass_matrix"]

        _adapt_kernel_key = (
            "vmap_mcmc_adapt",
            self.compile_signature(),
            method,
            int(n_warmup),
            bool(use_dense),
            float(target_accept_rate),
            int(n_leapfrog_steps),
            int(n_dim),
        )
        _run_adapt = self._memo_batch_kernel(
            "_batch_adapt_kernel_cache",
            _adapt_kernel_key,
            lambda: jax.jit(_run_window_adaptation),
        )
        step_size, inv_mass_matrix = _run_adapt(adapt_key, init_flats[0], first_data_args)

        if verbose:
            logger.info(
                "  vmap %s: %d galaxies × %d samples (D=%d, step_size=%.4f)",
                method,
                n_gal,
                n_samples,
                n_dim,
                float(step_size),
            )

        # Raw scan functions (no @jax.jit — the outer jit+vmap handles it)
        if method == "mcmc_nuts":
            kernel = _get_nuts_kernel()

            def _sample_scan(state, keys, data_args_i, step_size, inv_mass_matrix):
                """Scan over MCMC steps for a single galaxy (NUTS variant).

                ``step_size`` / ``inv_mass_matrix`` are threaded in (not closed
                over) so the compiled kernel is independent of the adapted
                values — see :meth:`_fit_batch_vmap_mcmc`.
                """

                def ld(pos):
                    """Log-density for this galaxy.

                    Parameters
                    ----------
                    pos : ndarray, shape (n_dim,)
                        Flattened unbounded parameters.

                    Returns
                    -------
                    float
                        Log posterior density.

                    Notes
                    -----
                    Inner closure for NUTS kernel. Not part of public API.
                    """
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
                    """Execute one NUTS kernel step."""
                    s, info = kernel(
                        k,
                        s,
                        ld,
                        step_size,
                        inv_mass_matrix,
                        max_num_doublings,
                    )
                    return s, (s.position, info.is_divergent)

                return jax.lax.scan(_step, state, keys)

        elif method == "mcmc_hmc":
            kernel = _get_hmc_kernel()

            def _sample_scan(state, keys, data_args_i, step_size, inv_mass_matrix):
                """Scan over MCMC steps for a single galaxy (HMC variant).

                Adaptation (``step_size`` / ``inv_mass_matrix``) is threaded in,
                not closed over — see :meth:`_fit_batch_vmap_mcmc`.
                """

                def ld(pos):
                    """Log-density for this galaxy.

                    Parameters
                    ----------
                    pos : ndarray, shape (n_dim,)
                        Flattened unbounded parameters.

                    Returns
                    -------
                    float
                        Log posterior density.

                    Notes
                    -----
                    Inner closure for HMC kernel. Not part of public API.
                    """
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
                    """Execute one HMC kernel step."""
                    s, info = kernel(
                        k,
                        s,
                        ld,
                        step_size,
                        inv_mass_matrix,
                        n_leapfrog_steps,
                    )
                    return s, (s.position, info.is_divergent)

                return jax.lax.scan(_step, state, keys)

        elif method == "mcmc_dynamic_hmc":
            kernel = _get_dynamic_hmc_kernel()

            def _sample_scan(state, keys, data_args_i, step_size, inv_mass_matrix):
                """Scan over MCMC steps for a single galaxy (dynamic HMC variant).

                Adaptation (``step_size`` / ``inv_mass_matrix``) is threaded in,
                not closed over — see :meth:`_fit_batch_vmap_mcmc`.
                """

                def ld(pos):
                    """Log-density for this galaxy.

                    Parameters
                    ----------
                    pos : ndarray, shape (n_dim,)
                        Flattened unbounded parameters.

                    Returns
                    -------
                    float
                        Log posterior density.

                    Notes
                    -----
                    Inner closure for dynamic HMC kernel. Not part of public API.
                    """
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
                    """Execute one dynamic HMC kernel step."""
                    s, info = kernel(k, s, ld, step_size, inv_mass_matrix)
                    return s, (s.position, info.is_divergent)

                return jax.lax.scan(_step, state, keys)

        elif method == "mcmc_ghmc":
            kernel = _get_ghmc_kernel()

            def _sample_scan(state, keys, data_args_i, step_size, inv_mass_matrix):
                """Scan over MCMC steps for a single galaxy (GHMC variant).

                Adaptation is threaded in, not closed over — see
                :meth:`_fit_batch_vmap_mcmc`. The GHMC momentum scale is derived
                from the passed ``inv_mass_matrix``.
                """
                if inv_mass_matrix.ndim == 2:
                    momentum_inv_scale = jnp.sqrt(jnp.diag(inv_mass_matrix))
                else:
                    momentum_inv_scale = jnp.sqrt(inv_mass_matrix)

                def ld(pos):
                    """Log-density for this galaxy.

                    Parameters
                    ----------
                    pos : ndarray, shape (n_dim,)
                        Flattened unbounded parameters.

                    Returns
                    -------
                    float
                        Log posterior density.

                    Notes
                    -----
                    Inner closure for GHMC kernel. Not part of public API.
                    """
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
                    """Execute one GHMC kernel step."""
                    s, info = kernel(
                        k,
                        s,
                        ld,
                        step_size,
                        momentum_inv_scale,
                        alpha,
                        delta,
                    )
                    return s, (s.position, info.is_divergent)

                return jax.lax.scan(_step, state, keys)

        # Single-galaxy function to vmap
        def single_galaxy(gal_key, init_flat_i, data_args_i, step_size, inv_mass_matrix):
            """Run inference (warmup + sampling) for a single galaxy.

            Parameters
            ----------
            gal_key : jax.random.PRNGKey
                Random seed for this galaxy.
            init_flat_i : ndarray, shape (n_dim,)
                Initial flattened unbounded parameters.
            data_args_i : dict
                Data arguments (fluxes, noise) for this galaxy.

            Returns
            -------
            tuple of (ndarray, ndarray)
                Posterior samples and (optionally) divergence indicators.

            Notes
            -----
            Designed for use with ``jax.vmap`` in batch MCMC. Captures
            ``kernel``, ``_sample_scan`` from outer scope; the adapted
            ``step_size`` / ``inv_mass_matrix`` are passed as arguments
            (broadcast via the vmap ``in_axes``) so the compiled kernel is
            reusable across ``fit_batch`` calls. Not part of public API.
            """

            def ld(pos):
                """Log-density for this galaxy.

                Parameters
                ----------
                pos : ndarray, shape (n_dim,)
                    Flattened unbounded parameters.

                Returns
                -------
                float
                    Log posterior density.

                Notes
                -----
                Inner closure. Not part of public API.
                """
                return logdensity_flat_2arg(pos, data_args_i)

            init_key, burn_key, sample_key = jax.random.split(gal_key, 3)

            if method == "mcmc_ghmc":
                # Keyword args: blackjax reordered ghmc.init's (rng_key, logdensity_fn)
                # between 1.3 and 1.6 — keywords are correct on both. (The single-galaxy
                # GHMC paths were already fixed this way; this batch path was missed
                # because it had no test.)
                state = blackjax.mcmc.ghmc.init(
                    position=init_flat_i, logdensity_fn=ld, rng_key=init_key
                )
            elif method == "mcmc_hmc":
                state = blackjax.mcmc.hmc.init(init_flat_i, ld)
            elif method == "mcmc_dynamic_hmc":
                state = blackjax.mcmc.dynamic_hmc.init(init_flat_i, ld, init_key)
            else:
                state = blackjax.mcmc.nuts.init(init_flat_i, ld)

            # Burn-in (discarded)
            if n_burnin > 0:
                burnin_keys = jax.random.split(burn_key, n_burnin)
                state, _ = _sample_scan(
                    state, burnin_keys, data_args_i, step_size, inv_mass_matrix
                )

            # Sampling
            sample_keys = jax.random.split(sample_key, n_samples)
            _, (positions, divergent) = _sample_scan(
                state, sample_keys, data_args_i, step_size, inv_mass_matrix
            )
            return positions, divergent

        # vmap + jit: one XLA kernel for all galaxies. The adapted step_size /
        # inv_mass_matrix are broadcast (in_axes None) so they enter as
        # Parameter ops, not baked Constants — which lets the memo below reuse
        # the compiled sampler across repeated same-config fit_batch calls
        # (a fresh closure would otherwise miss jax.jit's cache and recompile).
        gal_keys = jax.random.split(key, n_gal)
        _mcmc_kernel_key = (
            "vmap_mcmc",
            self.compile_signature(),
            method,
            int(n_burnin),
            int(n_samples),
            int(max_num_doublings),
            int(n_leapfrog_steps),
            float(alpha),
            float(delta),
            bool(use_dense),
            int(n_dim),
        )
        _run_batch = self._memo_batch_mcmc_kernel(
            _mcmc_kernel_key,
            lambda: jax.jit(jax.vmap(single_galaxy, in_axes=(0, 0, 0, None, None))),
        )
        all_positions, all_divergent = _run_batch(
            gal_keys,
            init_flats,
            batch_data_args,
            step_size,
            inv_mass_matrix,
        )

        t_sample = time.time() - t0

        if verbose:
            total_div = int(jnp.sum(all_divergent))
            logger.info(
                "  Done: %d galaxies in %.1fs (%.2fs/galaxy, %d divergences)",
                n_gal,
                t_sample,
                t_sample / n_gal,
                total_div,
            )

        # Post-process: unravel flat positions to physical params
        results = []
        for g_idx in range(n_gal):
            positions_i = all_positions[g_idx]
            divergent_i = all_divergent[g_idx]
            samples_phys = _vmap_samples_to_physical(positions_i, unravel_fn, self._to_physical)
            best_params = _mean_params(samples_phys)
            n_div = int(jnp.sum(divergent_i))
            result_i = Posterior(
                samples=samples_phys,
                params=best_params,
                method=f"{method} (vmap)",
                wall_time_s=t_sample / n_gal,
                diagnostics={
                    "n_warmup": n_warmup,
                    "n_burnin": n_burnin,
                    "n_samples": n_samples,
                    "n_divergent": n_div,
                    "step_size": float(step_size),
                    "batch_size": n_gal,
                },
                _model=self.model,
            )
            results.append(result_i)

        return results

    def _memo_batch_map_kernel(self, key, builder):
        """Cache a ``jax.jit(jax.vmap(...))`` batch-MAP kernel on this Fitter.

        The vmapped optimizer kernel is a *fresh* closure on every
        :meth:`fit_batch` call, and a fresh function object misses
        ``jax.jit``'s compilation cache — so a catalog processed in repeated
        same-shape ``fit_batch("map")`` calls recompiles the (expensive)
        vmapped loss each time. Memoizing the wrapper object under a key that
        fully determines the closure lets ``jax.jit`` reuse the compiled
        executable across calls.

        Safety: the observed data threads through as a *traced argument*
        (never the closure), and ``loss_fn`` is already structurally cached
        (:meth:`_get_or_build_loss_fn`), so the only closure-defining state is
        the optimizer configuration carried in ``key``. The cache lives on the
        Fitter instance, whose model structure is fixed at construction, so a
        cached kernel is only ever reused for the same model structure and the
        same optimizer config — per-galaxy data still flows in fresh through the
        traced ``batch_data_args``. ``key=None`` (e.g. a caller-supplied custom
        optimizer we cannot fingerprint) disables the memo — always build fresh
        rather than risk a stale kernel.

        Parameters
        ----------
        key : hashable or None
            Signature that fully determines the built kernel, or ``None`` to
            skip caching.
        builder : callable
            Zero-arg factory returning the ``jax.jit(jax.vmap(...))`` wrapper.

        Returns
        -------
        callable
            The (possibly cached) compiled batch kernel.
        """
        return self._memo_batch_kernel("_batch_map_kernel_cache", key, builder)

    def _memo_batch_mcmc_kernel(self, key, builder):
        """Cache the ``jax.jit(jax.vmap(single_galaxy))`` batch-MCMC sampler.

        Sibling of :meth:`_memo_batch_map_kernel` for the vmapped MCMC path
        (:meth:`_fit_batch_vmap_mcmc`). Same rationale and safety argument: the
        per-galaxy data AND the adapted ``step_size`` / ``inv_mass_matrix`` are
        threaded as arguments (the latter via ``jax.vmap(..., in_axes=(0, 0, 0,
        None, None))``), so the compiled sampler does not depend on their values
        and the memo key need only carry the structural sampler configuration
        (method, burn-in / sample counts, tree/leapfrog limits, GHMC α/δ, mass-
        matrix shape). ``key=None`` disables the memo.
        """
        return self._memo_batch_kernel("_batch_mcmc_kernel_cache", key, builder)

    #: Max compiled batch kernels retained per cache. A cached kernel can bake the
    #: SSP grid (~tens of MB), so an unbounded cache on a long-lived Fitter that
    #: sees many distinct configs would pin them all. The common case (one config,
    #: reused across a same-shape catalog) stays well under this.
    _BATCH_KERNEL_CACHE_MAX = 16

    def _memo_batch_kernel(self, cache_attr, key, builder):
        """Instance-scoped LRU memo for a compiled batch kernel (shared MAP/MCMC core).

        Returns ``builder()`` uncached when ``key`` is ``None``; otherwise caches
        the built wrapper in ``self.__dict__[cache_attr]`` keyed by ``key`` so
        ``jax.jit`` can reuse the compiled executable across ``fit_batch`` calls.
        The cache is bounded at :attr:`_BATCH_KERNEL_CACHE_MAX` (LRU eviction) so
        varying-config workloads cannot grow retention without limit.
        """
        if key is None:
            return builder()
        cache = self.__dict__.setdefault(cache_attr, {})
        fn = cache.get(key)
        if fn is None:
            fn = builder()
            cache[key] = fn
            if len(cache) > self._BATCH_KERNEL_CACHE_MAX:
                # dicts preserve insertion order — drop the oldest (least recent).
                cache.pop(next(iter(cache)))
        else:
            # LRU touch: reinsert so a hit becomes the most-recently-used entry.
            del cache[key]
            cache[key] = fn
        return fn

    def _fit_batch_vmap_map(self, batch, *, key, verbose=True, **kwargs):
        """Vectorized MAP optimization over a batch using jax.vmap.

        All galaxies share the same compiled XLA kernel — parameters and
        optimizer states are batched across the first axis. A single
        ``jax.jit(jax.vmap(step))`` call optimizes all galaxies in parallel.

        Requirements: same model (precomp set), same data shape per galaxy.
        """
        from tengri.inference.backends.map_dispatch import _JAXOPT_SOLVERS
        from tengri.inference.posterior import Posterior

        n_steps = kwargs.get("n_steps", 1000)
        learning_rate = kwargs.get("learning_rate", 0.02)
        optimizer = kwargs.get("optimizer", "adam")
        print_every = kwargs.get("print_every", 200)

        n_gal = len(batch)
        t0 = time.time()

        # Stack galaxy data into batch arrays (n_gal, n_obs)
        flux_batch = jnp.stack([jnp.asarray(g["flux_obs"]) for g in batch])
        noise_batch = jnp.stack([jnp.asarray(g["noise"]) for g in batch])
        batch_data_args = {
            "data": flux_batch,
            "noise": noise_batch,
            "sqrt_noise_inv": inv_noise_std(noise_batch),
        }

        # Initialize params for each galaxy independently
        init_keys = jax.random.split(key, n_gal)
        init_params_list = [self._initialize_unbounded(k) for k in init_keys]
        params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *init_params_list)

        loss_fn = self._get_or_build_loss_fn()

        # ── jaxopt quasi-Newton / line-search path ──
        if isinstance(optimizer, str) and optimizer in _JAXOPT_SOLVERS:
            from tengri.inference.backends.map_dispatch import _build_jaxopt_solver

            tol = kwargs.get("tol", 1e-5)
            solver, opt_name = _build_jaxopt_solver(
                optimizer,
                loss_fn,
                maxiter=n_steps,
                tol=tol,
            )

            if verbose:
                logger.info(
                    "  vmap MAP (%s): %d galaxies × %d max iter (single JIT kernel)",
                    opt_name,
                    n_gal,
                    n_steps,
                )

            run_kernel = self._memo_batch_map_kernel(
                ("jaxopt", optimizer, int(n_steps), float(tol)),
                lambda: jax.jit(jax.vmap(solver.run)),
            )
            batch_result = run_kernel(params_batch, batch_data_args)

            t_total = time.time() - t0
            if verbose:
                logger.info(
                    "  Done: %d galaxies in %.1fs (%.2fs/galaxy)",
                    n_gal,
                    t_total,
                    t_total / n_gal,
                )

            results = []
            for g_idx in range(n_gal):
                params_i = jax.tree.map(lambda x, idx=g_idx: x[idx], batch_result.params)
                bounded_i = self._to_physical(params_i)
                result_i = Posterior(
                    samples=None,
                    params=bounded_i,
                    method=f"map ({opt_name})",
                    wall_time_s=t_total,
                    diagnostics={
                        "loss": float(batch_result.state.value[g_idx]),
                        "n_steps": int(batch_result.state.iter_num[g_idx]),
                        "optimizer": opt_name,
                        "converged": bool(batch_result.state.error[g_idx] < tol),
                    },
                    _model=self.model,
                    _fitter=self,
                )
                results.append(result_i)

            return results

        # ── optax iterative path (adam / adamw / sgd / custom) ──
        try:
            import optax
        except ImportError:
            raise ImportError("optax required for MAP: pip install optax") from None

        if isinstance(optimizer, str):
            _opt_builders = {
                "adam": lambda: optax.adam(learning_rate),
                "adamw": lambda: optax.adamw(learning_rate),
                "sgd": lambda: optax.sgd(learning_rate, momentum=0.9),
            }
            opt = _opt_builders[optimizer]()
        else:
            opt = optimizer

        opt_states_batch = jax.vmap(opt.init)(params_batch)

        def single_step(params, opt_state, data_args_i):
            """Perform one optimization step for a single galaxy.

            Parameters
            ----------
            params : ndarray, shape (n_dim,)
                Flattened unbounded parameters for this galaxy.
            opt_state : optax.OptState
                Optimizer state (e.g., Adam momentum buffers).
            data_args_i : dict
                Data arguments (fluxes, noise) for this galaxy.

            Returns
            -------
            tuple of (ndarray, optax.OptState, float)
                Updated parameters, optimizer state, and loss scalar.

            Notes
            -----
            Designed for use with ``jax.vmap`` in batch MAP. Captures
            ``loss_fn``, ``opt`` from outer scope. Not part of public API.
            """
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, data_args_i))(params)
            updates, new_opt_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        # Memoize the vmapped step so repeated same-config fit_batch("map")
        # calls reuse the compiled kernel (a fresh closure would miss jax.jit's
        # cache). Only string optimizers are fingerprintable; a caller-supplied
        # custom optax object disables the memo (build fresh).
        _opt_memo_key = (
            ("optax", optimizer, float(learning_rate)) if isinstance(optimizer, str) else None
        )
        batch_step = self._memo_batch_map_kernel(
            _opt_memo_key, lambda: jax.jit(jax.vmap(single_step))
        )

        params = params_batch
        opt_states = opt_states_batch

        if verbose:
            logger.info("  vmap MAP: %d galaxies × %d steps (single JIT kernel)", n_gal, n_steps)

        for i in range(n_steps):
            params, opt_states, losses = batch_step(params, opt_states, batch_data_args)
            if verbose and (i % print_every == 0 or i == n_steps - 1):
                mean_loss = float(losses.mean())
                logger.info("  Step %5d/%d: mean loss = %.4f", i, n_steps, mean_loss)

        t_total = time.time() - t0
        if verbose:
            logger.info(
                "  Done: %d galaxies in %.1fs (%.2fs/galaxy)",
                n_gal,
                t_total,
                t_total / n_gal,
            )

        results = []
        for g_idx in range(n_gal):
            params_i = jax.tree.map(lambda x, idx=g_idx: x[idx], params)
            bounded_i = self._to_physical(params_i)
            result_i = Posterior(
                samples=None,
                params=bounded_i,
                method="map",
                wall_time_s=t_total,
                diagnostics={"loss": float(losses[g_idx]), "n_steps": n_steps},
                _model=self.model,
                _fitter=self,
            )
            results.append(result_i)

        return results


# ── Backend Registry Initialization ──────────────────────────────────────────
# All ``@register_backend(...)`` calls live in ``inference/_registration.py``.
# That module is imported for its side effects by ``inference/__init__.py``,
# which guarantees the registry is populated before any caller can dispatch
# through ``Fitter.run``. See ADR-0010.
