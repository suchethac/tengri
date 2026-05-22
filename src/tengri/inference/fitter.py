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
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

__all__ = ["Fitter", "resolve_method"]

if TYPE_CHECKING:
    from tengri.inference.posterior import Posterior

import jax

logger = logging.getLogger(__name__)
import jax.numpy as jnp

from tengri.config.exceptions import ParameterError
from tengri.inference._model_cache import get_model_cache
from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical
from tengri.inference.jit_engine import build_jit_engine
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


def _maybe_population_delegate(model):
    """Return a configured :class:`PopulationFitter` when ``model`` is hierarchical.

    Detects the case where the user built a ``ForwardModel`` with
    ``ForwardModel.build(population=PopulationSEDModel(...))``.
    When that's true, constructs the :class:`PopulationFitter`
    instance that will drive the hierarchical inference: the SED
    template is rebuilt with the shared parameters fixed at each
    inference call, the per-galaxy data lives on the population's
    ``galaxies`` list, and the shared priors come from
    ``PopulationSEDModel.priors``.

    Returns ``None`` for the standard (single-galaxy, multi-pop
    decomposition, spatial-only) cases — the caller continues with
    the regular Fitter init.

    Parameters
    ----------
    model : Any
        The ``model`` argument passed to ``Fitter.__init__``.

    Returns
    -------
    PopulationFitter or None
        Configured delegate; ``None`` if ``model`` is not a
        ForwardModel-with-PopulationSEDModel.
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
    pop = populations[0]
    pop_sed = getattr(pop, "sed", None)
    if not isinstance(pop_sed, PopulationSEDModel):
        return None

    # Today's PopulationFitter only knows the two PSD shared parameters
    # by name. If the user set ``shared=`` to something else, surface
    # a clear NotImplementedError rather than silently using the wrong
    # priors.
    expected_shared = ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr")
    if pop_sed.shared != expected_shared:
        raise NotImplementedError(
            "Fitter routing for PopulationSEDModel currently supports only the "
            f"default shared parameters {expected_shared}. PopulationSEDModel "
            f"with shared={pop_sed.shared!r} needs a generalised "
            "PopulationFitter — tracked in issue #211."
        )

    from tengri.inference.hierarchical import PopulationFitter

    # Build the (psd_sigma, psd_tau_myr) -> SEDModel factory expected
    # by PopulationFitter from the SED template.
    sed_template = pop_sed.sed
    factory = _build_population_factory(sed_template)
    psd_sigma_prior = pop_sed.priors.get("sfh_field_psd_sigma", (0.1, 4.0))
    psd_tau_prior = pop_sed.priors.get("sfh_field_psd_tau_myr", (1.0, 300.0))

    return PopulationFitter(
        factory,
        list(pop_sed.galaxies),
        psd_sigma_prior=psd_sigma_prior,
        psd_tau_prior=psd_tau_prior,
        data_type=pop_sed.data_type,
        _via_routing=True,
    )


def _build_population_factory(sed_template):
    """Build a ``(psd_sigma, psd_tau_myr) -> SEDModel`` closure.

    The closure clones the SED template with the shared PSD parameters
    fixed to the provided values. Used internally by
    :func:`_maybe_population_delegate` to bridge
    :class:`PopulationSEDModel` (the new SubModel construction shape)
    to :class:`PopulationFitter` (the existing inference machinery).
    """

    def factory(psd_sigma, psd_tau_myr):
        overrides = {
            "sfh_field_psd_sigma": psd_sigma,
            "sfh_field_psd_tau_myr": psd_tau_myr,
        }
        if hasattr(sed_template, "with_fixed"):
            return sed_template.with_fixed(**overrides)

        # Fallback: rebuild via SEDModel constructor with the spec mutated.
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.priors import Fixed

        spec = sed_template.spec
        spec_kwargs = dict(getattr(spec, "kwargs", {}))
        for name, value in overrides.items():
            spec_kwargs[name] = Fixed(value)
        new_spec = type(spec)(**spec_kwargs)
        ssp_data = getattr(sed_template, "ssp_data", None)
        observation = getattr(sed_template, "observation", None)
        if ssp_data is None or observation is None:
            raise RuntimeError(
                "Cannot rebuild SEDModel from template: missing ssp_data or "
                "observation. Add SEDModel.with_fixed for a clean path."
            )
        return SEDModel(new_spec, ssp_data, observation=observation)

    return factory


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
    data_mask : array_like, bool or None
        Optional boolean mask for censored/non-detections. ``True`` = use datum
        in likelihood, ``False`` = exclude. Default ``None`` (use all).
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
    Fit a single galaxy with geoVI (default):

    >>> from tengri import SEDModel, Fitter, Parameters
    >>> model = SEDModel(Parameters())
    >>> data = jnp.array([1.2, 0.8, 0.5])  # photometric fluxes
    >>> noise = jnp.array([0.1, 0.08, 0.06])
    >>> fitter = Fitter(model, data, noise)
    >>> result = fitter.run("vi", n_samples=100)
    >>> print(result.params)

    Fit with warm-start from MAP:

    >>> result_map = fitter.run("map", n_steps=1000)
    >>> result_mcmc = fitter.run("mcmc_nuts", init_from=result_map, n_warmup=500)

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
    ):
        # ── Hierarchical-population routing (issue #211) ────────────
        # When ``model`` is a ForwardModel whose SubModel is a
        # PopulationSEDModel, the per-galaxy data already lives on the
        # population (in pop.galaxies). Route inference through the
        # existing PopulationFitter machinery; the user-facing
        # Fitter(forward).run('vi') call stays uniform.
        self._population_delegate = _maybe_population_delegate(model)
        if self._population_delegate is not None:
            self.model = model
            # Surface the SED-template spec so callers that inspect
            # fitter.spec still see something meaningful.
            self.spec = getattr(self._population_delegate, "_spec", None)
            self._user_likelihood = likelihood
            return

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
        # marginalisation are NOT applied automatically — wrap them
        # into the user likelihood if needed.
        self._user_likelihood = likelihood
        self._auto_protocol_likelihood = auto_protocol_likelihood

        # ── Orchestrator opt-in (2026-05) ───────────────────────────
        # When True, route forward predictions through
        # :meth:`SEDModel.predict_state` (the SEDComponent
        # chain) instead of the legacy fused ``predict_photometry`` /
        # ``predict_spectrum`` kernels. Default ``False`` preserves
        # existing inference behaviour bit-for-bit. Spectroscopy has no
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
        self.data_mask = jnp.asarray(data_mask) if data_mask is not None else None
        self.data_type = self._resolve_data_type(data_type, model)
        self.spec = model.spec

        if self.use_components and self.data_type not in ("photometry", "spectroscopy", "joint"):
            raise NotImplementedError(
                "Fitter(use_components=True) currently supports "
                f"data_type in (photometry, spectroscopy, joint); got {self.data_type!r}."
            )

        # ── Auto-precompute photometry ─────────────────────────────
        self._auto_precompute_photometry(model)

        # ── Calibration ────────────────────────────────────────────
        self._calibration_marginalize = calibration_marginalize
        self._cal_n_poly = cal_n_poly
        self._cal_prior_sigma = cal_prior_sigma
        self._has_spectroscopy = self.data_type in ("spectroscopy", "joint")

        # ── Emission lines ─────────────────────────────────────────
        self._init_emission_lines(model, eline_marginalize, eline_prior_type)

        # ── Parameters ─────────────────────────────────────────────
        self._free_names = self.spec.free_params
        self._fixed_values = self.spec.get_fixed_values()
        self._bounds = {n: self.spec.get_distribution(n).bounds for n in self._free_names}

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
        - calibration marginalisation → ``CalibrationMarginalisedLikelihood``
        - flat-prior e-line marginalisation → ``ELineMarginalisedLikelihood``
          (with a per-call design-matrix builder closure)
        - line fluxes / spectral indices → composed onto the base
          via ``CompositeLikelihood``

        Returns ``None`` only for cases the Protocol path does not yet
        cover — currently the Cloudy-prior e-line marginalisation
        (uses a different math primitive) and the e-line *fitted*
        amplitudes (line amplitudes are fit, not marginalised).
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

    def _auto_precompute_photometry(self, model: Any) -> None:
        """Auto-trigger photometry precomputation if conditions are met.

        Fires when: fixed redshift + filters present + not yet precomputed.
        Lets users create a Model without ``precompute=True`` and still get
        the fast fused path when they construct a Fitter.
        """
        if (
            self.data_type not in ("photometry", "joint")
            or model._precomputed.photometry is not None
            or getattr(model, "_z_fixed", None) is None
            or getattr(model, "filter_waves", None) is None
        ):
            return

        import contextlib

        from tengri.components.stellar.sps.precompute import precompute_photometry

        model._precomputed.photometry = precompute_photometry(
            model.ssp_data,
            model.filter_waves,
            model.filter_trans,
            model._z_fixed,
            model._dl_cm_fixed,
        )
        with contextlib.suppress(Exception):
            model._hybrid = model._build_hybrid_kernels()

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
            self.spec = self.spec.merge_observation_params(**_amp_priors)
        else:
            self._eline_amplitude_names = []

    def _build_data_args(self, model: Any) -> dict:
        """Build the data-dependent argument dict passed to JIT'd loss functions.

        These are passed as explicit arguments (not closed over) so that
        engines compiled for one galaxy can be reused for another with
        the same model + parameter structure.
        """
        noise_inv = 1.0 / self.noise**2
        args = {
            "data": self.data,
            "noise": self.noise,
            "noise_inv": noise_inv,
            "sqrt_noise_inv": jnp.sqrt(noise_inv),
            "n_data": jnp.int32(len(self.data)),
        }
        if self.data_mask is not None:
            args["data_mask"] = self.data_mask

        obs = getattr(model, "observation", None)
        if obs is not None:
            spec_cfg = getattr(obs, "spectroscopy", None)
            if spec_cfg is not None and getattr(spec_cfg, "has_covariance", False):
                args["spec_cov_inv"] = spec_cfg.cov_inv

            line_flux_cfg = getattr(obs, "line_fluxes", None)
            if line_flux_cfg is not None:
                args["line_flux_obs"] = line_flux_cfg.fluxes
                args["line_flux_err"] = line_flux_cfg.errors
                args["line_flux_waves"] = line_flux_cfg.wavelengths

            index_cfg = getattr(obs, "spectral_indices", None)
            if index_cfg is not None:
                args["index_obs"] = index_cfg.values
                args["index_err"] = index_cfg.errors

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
        from tengri.observation.noise import has_noise_model

        model_sig = self.model.compile_signature()
        fitter_sig = (
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
        )
        return (model_sig, fitter_sig)

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
        flag, latent dimension, data length, free parameter names, and
        noise model presence).
        """
        from tengri.observation.noise import has_noise_model

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
        )

    def _get_or_build_engine(self, pos_dict: dict) -> dict:
        """Return the JIT engine, reusing a cached version when possible.

        Engines are cached in a module-level shared cache keyed by
        compile_signature(), enabling zero-recompile fits when multiple
        Fitters share the same model structure (e.g., catalog fits with
        different SSP files of identical shape).

        Also maintains a backward-compat per-model cache for any code
        that reads directly from get_model_cache(self.model)["jit_engine"].

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
        per_model_cache = get_model_cache(self.model).setdefault("jit_engine", {})
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
        >>> fitter = Fitter(model, data, noise)
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
                        10,
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

    def _build_loss_fn(self, mode: str = "traced") -> Callable:
        """Build a differentiable loss function.

        See ``tengri.inference.loss_functions.build_loss_fn`` for full docs.
        Returns ``loss_fn(params_unbounded, data_args) -> scalar``.

        Parameters
        ----------
        mode : str, optional
            Forward model prediction mode. Default "traced" is for
            internal tracing mode (NIFTy VI path). Use "auto" for ~1.5x speedup with
            non-NIFTy methods.
        """
        return build_loss_fn(self, mode=mode)

    def _get_or_build_loss_fn(self, mode: str = "traced") -> Callable:
        """Return the cached loss function, building it if needed.

        The loss function is cached on the Model object keyed by
        ``_engine_cache_key()`` + mode so that multiple Fitters with the same
        model structure share the same compiled XLA program.

        Parameters
        ----------
        mode : str, optional
            Forward model prediction mode. Default "traced" for backward
            compatibility. Pass "auto" for ~1.5x speedup with non-NIFTy methods.
        """
        from tengri.inference.jit_engine import get_or_build_cached

        cache_key = (self._engine_cache_key(), mode)
        per_model = get_model_cache(self.model).setdefault("loss_fn", {})
        if cache_key in per_model:
            return per_model[cache_key]
        loss_fn = get_or_build_cached(self, mode, "loss", lambda: self._build_loss_fn(mode=mode))
        per_model[cache_key] = loss_fn
        return loss_fn

    def _build_logprior_fn(self) -> Callable:
        """Build a log-prior function. See ``loss_functions.build_logprior_fn``."""
        return build_logprior_fn(self)

    def _build_loglikelihood_fn(self, mode: str = "traced") -> Callable:
        """Build log-likelihood function. See ``loss_functions.build_loglikelihood_fn``."""
        return build_loglikelihood_fn(self, mode=mode)

    def _get_or_build_loglikelihood_fn(self, mode: str = "traced") -> Callable:
        """Return the cached log-likelihood function, building if needed."""
        from tengri.inference.jit_engine import get_or_build_cached

        cache_key = (self._engine_cache_key(), mode)
        per_model = get_model_cache(self.model).setdefault("loglik_fn", {})
        if cache_key in per_model:
            return per_model[cache_key]
        loglik_fn = get_or_build_cached(
            self, mode, "loglik", lambda: self._build_loglikelihood_fn(mode=mode)
        )
        per_model[cache_key] = loglik_fn
        return loglik_fn

    def _build_loglikelihood_unbounded_fn(self, mode: str = "traced") -> Callable:
        """Build unbounded-space log-likelihood.

        See ``loss_functions.build_loglikelihood_unbounded_fn``.
        """
        return build_loglikelihood_unbounded_fn(self, mode=mode)

    def _get_or_build_grad_fn(self, mode: str = "traced") -> Callable:
        """Return cached JIT-compiled value_and_grad of the loss function.

        The gradient function takes ``(params_unbounded, data_args)`` as
        explicit arguments so the compiled XLA program is reusable across
        galaxies with the same model structure.
        """
        from tengri.inference.jit_engine import get_or_build_cached

        cache_key = (self._engine_cache_key(), mode)
        per_model = get_model_cache(self.model).setdefault("grad_fn", {})
        if cache_key in per_model:
            return per_model[cache_key]

        loss_fn = self._get_or_build_loss_fn(mode=mode)

        def _build():
            @jax.jit
            def val_and_grad(params_u, data_args):
                """Loss and gradient w.r.t. unbounded parameters."""
                return jax.value_and_grad(lambda p: loss_fn(p, data_args))(params_u)

            return val_and_grad

        val_and_grad = get_or_build_cached(self, mode, "grad", _build)
        per_model[cache_key] = val_and_grad
        return val_and_grad

    def _get_or_build_logdensity_fn(self, mode: str = "traced") -> Callable:
        """Return cached JIT-compiled log-density for MCMC/Pathfinder.

        Returns ``logdensity(params_u, data_args) -> scalar``.  Callers
        should partial-apply ``data_args`` for blackjax compatibility.
        """
        cache_key = (self._engine_cache_key(), mode)
        from tengri.inference.jit_engine import get_or_build_cached

        per_model = get_model_cache(self.model).setdefault("logdensity_fn", {})
        if cache_key in per_model:
            return per_model[cache_key]

        loss_fn = self._get_or_build_loss_fn(mode=mode)

        def _build():
            @jax.jit
            def logdensity(params_u, data_args):
                """Log posterior (negative loss) for MCMC."""
                return -loss_fn(params_u, data_args)

            return logdensity

        logdensity = get_or_build_cached(self, mode, "logdensity", _build)
        per_model[cache_key] = logdensity
        return logdensity

    # ── Parameter transforms ──────────────────────────────────────────

    def _initialize_unbounded(self, key: Any) -> dict:
        """Create initial unbounded parameter dict."""
        params = {}
        keys = jax.random.split(key, len(self._free_names) + 1)

        for i, name in enumerate(self._free_names):
            dist = self.spec.get_distribution(name)
            if isinstance(dist, Gaussian):
                params[name] = dist.standardize(jnp.array(dist.mu))
            else:
                # Initialize near midpoint (u=0) with small perturbation
                params[name] = 0.1 * jax.random.normal(keys[i])

        if self.spec.stochastic:
            params["psd_xi"] = 0.1 * jax.random.normal(keys[-1], shape=(self.spec.n_grid,))

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
            params[name] = jnp.array(val)
        if self.spec.stochastic and "psd_xi" in params_unbounded:
            params["psd_xi"] = params_unbounded["psd_xi"]
        return params

    # ── Inference dispatch ────────────────────────────────────────────

    def run(self, method: str = "vi_nonlinear_fast", *, init_from=None, key=None, **kwargs):
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

            - ``"vi"`` — geoVI via NIFTy (nonlinear, default for D>20)
            - ``"vi_nonlinear"`` — geoVI via NIFTy (alias of ``vi``)
            - ``"vi_linear"`` — MGVI via NIFTy (linearized Gaussian)
            - ``"vi_nonlinear_fast"`` — geoVI fast path (~35% faster, no logging)
            - ``"vi_linear_fast"`` — MGVI fast path (~35% faster, no logging)
            - ``"native_vi_nonlinear"`` — Native JAX geoVI (experimental; ~19× faster than NIFTy)
            - ``"native_vi_linear"`` — Native JAX MGVI (experimental)

            **MCMC Sampling**

            - ``"mcmc_nuts"`` — NUTS via BlackJAX (default for D≤20; exact posterior)
            - ``"mcmc_raytrace"`` — Ray Tracing (Behroozi 2025; O(1) gradient cost)
            - ``"mcmc"`` — Auto: NUTS (D≤20) or Ray Tracing (D>20)
            - ``"mcmc_hmc"`` — Standard HMC (fixed trajectory length)
            - ``"mcmc_dynamic_hmc"`` — Dynamic HMC (adaptive trajectory)
            - ``"mcmc_ghmc"`` — Generalized HMC (partial momentum refresh)
            - ``"mcmc_mclmc"`` — MCLMC (O(1) grad/sample, biased)
            - ``"mcmc_adjusted_mclmc"`` — MCLMC + Metropolis correction
            - ``"mcmc_ess"`` — Elliptical Slice Sampling (gradient-free)

            **Point Estimation & Approximations**

            - ``"map"`` — MAP optimization (Adam by default)
            - ``"laplace"`` — Laplace approximation (Gaussian posterior at MAP)
            - ``"pathfinder"`` — L-BFGS trajectory + sequence of Gaussians (Zhang+2022)

            **Model Comparison (Bayesian Evidence)**

            - ``"nss"`` — Nested Slice Sampling (exact Z, D≤30)

            **Automatic Selection**

            - ``"auto"`` — NUTS (D≤20) or geoVI (D>20) based on dimensionality

        init_from : Posterior, optional
            Previous inference result to use as warm-start initialization.
            The posterior mean is extracted and converted to unbounded space.
            Useful for refining results across different methods. Default ``None``.

        key : PRNGKey, optional
            JAX random key. Default ``PRNGKey(42)`` for reproducibility.
            Ignored for deterministic methods (``"map"``, ``"laplace"``).

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
          :doc:`bench/reports/2026-04-17_native_vs_nifty.md`.

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

        **Compile-cache behaviour (smart lean, 2026-05):**

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

        >>> fitter = Fitter(model, data, noise)
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
        # ── Hierarchical-population delegation (issue #211) ─────────
        # Constructed via Fitter(forward) where forward.populations[0].sed
        # is a PopulationSEDModel. PopulationFitter.run owns the
        # hierarchical machinery; this Fitter just delegates.
        if getattr(self, "_population_delegate", None) is not None:
            return self._population_delegate.run(method, key=key, **kwargs)

        if key is None:
            key = jax.random.PRNGKey(42)

        # Resolve deprecated aliases and validate method
        method = resolve_method(method)

        # --- Smart lean: drop only stale L3 entries before this run ---
        # Default: lean=True. The smart-lean path (below) preserves the
        # entry whose key matches this fitter's compile_signature, so
        # CatalogFitter loops and repeated identical fits hit the cache
        # without any opt-in. ``tengri.persistent()`` is rarely needed —
        # it only matters if you want to keep *non-matching* entries
        # alive (e.g. swapping back and forth between MAP and HMC and
        # wanting both compiles in RAM). Override per-call via
        # ``fitter.run(..., lean=True/False)``.
        from tengri.inference.jit_engine import (
            clear_shared_caches as _clear_shared_caches,
            is_lean_mode as _is_lean_mode,
            is_persistent_mode as _is_persistent_mode,
        )

        _user_lean = kwargs.pop("lean", None)
        if _user_lean is None:
            if _is_persistent_mode():
                _user_lean = False
            elif _is_lean_mode():
                _user_lean = True
            else:
                _user_lean = True
        if _user_lean:
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
            _clear_shared_caches(scope="inference_body", keep_sig=self._lean_keep_sig)

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

        # --- "auto" method: dimensionality-based selection ---
        if method == "auto":
            d = self.spec.n_free
            try:
                from tengri.parameters.defaults import get_inference_defaults

                threshold = int(get_inference_defaults().get("mcmc_auto_d", _AUTO_D_THRESHOLD))
            except (ImportError, FileNotFoundError, OSError, KeyError, ValueError):
                # Config unavailable, missing key, or non-integer value — use hardcoded fallback
                threshold = _AUTO_D_THRESHOLD
            method = "mcmc_nuts" if d <= threshold else "vi_nonlinear_fast"

        # --- Dispatch to underlying _run_* methods via registry ---
        from tengri.inference._backend_registry import check_requires, get_backend

        if method == "auto":
            # Pre-registry semantics: low-D → NUTS (exact), high-D → geoVI (scalable).
            # Dimensionality threshold is configurable via inference defaults.
            d = self.spec.n_free
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
            d = self.spec.n_free
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

        # Friendly error if the backend's optional dependency is missing,
        # before we descend into a deep third-party traceback.
        check_requires(entry)

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
        >>> fitter = Fitter(model, data, noise)
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
        n_grid = self.model._n_grid if self.model._uses_stochastic_sfh else 0
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

    def _get_mode_for_method(self, method: str) -> str:
        """Determine forward model prediction mode based on inference method.

        PERFORMANCE NOTE (2026-04-18): Profiling shows mode="traced" is
        12.64x FASTER than mode="auto" (5.9ms vs 74.4ms) with stable timing.
        mode="auto" has pathological variance (std=504ms, 6.8x the mean) causing
        occasional 500ms+ outliers. Always use mode="traced" for inference.

        Parameters
        ----------
        method : str
            Inference method name (e.g., "vi", "mcmc_nuts", "map")

        Returns
        -------
        str
            Always returns "traced" for optimal performance across all
            inference methods. Previous "auto" mode had severe variance issues.

        See Also
        --------
        docs/dev/jit-optimization-report-2026-04-18.md : Full profiling analysis
        """
        # ALL methods now use traced for 12.64x speedup + stable timing
        # (mode="auto" variance pathology fixed 2026-04-18)
        return "traced"

    # ── Private method runners ────────────────────────────────────────

    def _run_vi(self, *, key, init_from=None, **kwargs) -> Posterior:
        """Dispatch to geometric variational inference via NIFTy (nonlinear)."""
        from tengri.inference.backends.vi.nifty import run_nifty_vi

        kwargs.setdefault("sample_mode", "nonlinear_resample")
        return run_nifty_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_vi_linear(self, *, key, init_from=None, **kwargs) -> Posterior:
        """Dispatch to metric Gaussian variational inference via NIFTy (linear)."""
        from tengri.inference.backends.vi.nifty import run_nifty_vi

        kwargs.setdefault("sample_mode", "linear_resample")
        return run_nifty_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_vi_native(self, *, key, init_from=None, **kwargs) -> Posterior:
        """Dispatch to native JAX geometric variational inference (experimental)."""
        from tengri.inference.backends.vi.native import run_native_vi

        kwargs.setdefault("sample_mode", "geovi")
        return run_native_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_vi_native_linear(self, *, key, init_from=None, **kwargs) -> Posterior:
        """Dispatch to native JAX MGVI inference (experimental)."""
        from tengri.inference.backends.vi.native import run_native_vi

        kwargs.setdefault("sample_mode", "linear")
        return run_native_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_nifty_fast_vi(self, *, key, init_from=None, **kwargs) -> Posterior:
        """Dispatch to fast geoVI via NIFTy OptimizeVI (no logging, ~35% speedup)."""
        from tengri.inference.backends.vi.nifty import run_nifty_fast_vi

        kwargs.setdefault("sample_mode", "nonlinear_resample")
        return run_nifty_fast_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_nifty_fast_vi_linear(self, *, key, init_from=None, **kwargs) -> Posterior:
        """Dispatch to fast MGVI via NIFTy OptimizeVI (no logging, ~35% speedup)."""
        from tengri.inference.backends.vi.nifty import run_nifty_fast_vi

        kwargs.setdefault("sample_mode", "linear_resample")
        return run_nifty_fast_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_nss(self, *, key, **kwargs) -> Posterior:
        """Dispatch to nested slice sampling for Bayesian evidence estimation."""
        from tengri.inference.backends.evidence import run_nss

        return run_nss(self, key=key, **kwargs)

    def _run_map(self, *, key, **kwargs) -> Posterior:
        """Dispatch to MAP optimization via gradient descent (Adam by default)."""
        from tengri.inference.backends.map_dispatch import run_map

        return run_map(self, key=key, **kwargs)

    def _run_raytrace(self, *, key, **kwargs) -> Posterior:
        """Dispatch to Ray Tracing MCMC sampler (Behroozi 2025)."""
        from tengri.inference.backends.mcmc import run_raytrace

        return run_raytrace(self, key=key, **kwargs)

    def _run_nuts(self, *, key, **kwargs) -> Posterior:
        """Dispatch to NUTS sampler via BlackJAX for exact posterior sampling."""
        from tengri.inference.backends.mcmc import run_nuts

        return run_nuts(self, key=key, **kwargs)

    def _run_hmc(self, *, key, **kwargs) -> Posterior:
        """Dispatch to standard Hamiltonian Monte Carlo (fixed trajectory length)."""
        from tengri.inference.backends.mcmc import run_hmc

        return run_hmc(self, key=key, **kwargs)

    def _run_dynamic_hmc(self, *, key, **kwargs) -> Posterior:
        """Dispatch to dynamic HMC with adaptive trajectory length."""
        from tengri.inference.backends.mcmc import run_dynamic_hmc

        return run_dynamic_hmc(self, key=key, **kwargs)

    def _run_ghmc(self, *, key, **kwargs) -> Posterior:
        """Dispatch to generalized HMC with partial momentum refresh."""
        from tengri.inference.backends.mcmc import run_ghmc

        return run_ghmc(self, key=key, **kwargs)

    def _run_mclmc(self, *, key, **kwargs) -> Posterior:
        """Dispatch to micro-canonical Langevin MCMC (microcanonical dynamics)."""
        from tengri.inference.backends.mcmc import run_mclmc

        return run_mclmc(self, key=key, **kwargs)

    def _run_adjusted_mclmc(self, *, key, **kwargs) -> Posterior:
        """Dispatch to MCLMC with Metropolis-Hastings correction."""
        from tengri.inference.backends.mcmc import run_adjusted_mclmc

        return run_adjusted_mclmc(self, key=key, **kwargs)

    def _run_laplace(self, *, key, **kwargs) -> Posterior:
        """Dispatch to Laplace approximation (Gaussian posterior at MAP)."""
        from tengri.inference.backends.map_dispatch import run_laplace

        return run_laplace(self, key=key, **kwargs)

    def _run_pathfinder(self, *, key, **kwargs) -> Posterior:
        """Dispatch to Pathfinder (L-BFGS trajectory + best Gaussian fit)."""
        from tengri.inference.backends.map_dispatch import run_pathfinder

        return run_pathfinder(self, key=key, **kwargs)

    def _run_elliptical_slice(self, *, key, **kwargs) -> Posterior:
        """Dispatch to elliptical slice sampling (gradient-free)."""
        from tengri.inference.backends.mcmc.elliptical_slice import (
            run_elliptical_slice_fitter,
        )

        return run_elliptical_slice_fitter(self, key=key, **kwargs)

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
        #   4. otherwise unchunked (preserves prior behaviour)
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
        import blackjax

        if verbose:
            logger.info("  Drawing %d posterior samples via BlackJAX NUTS...", n_samples)

        if likelihood is not None:

            @jax.jit
            def logdensity_fn(x):
                """Evaluate negative log density from custom likelihood and standard prior.

                Parameters
                ----------
                x : dict
                    Parameter dict in unbounded space.

                Returns
                -------
                float
                    Log posterior (likelihood + Gaussian prior).

                Notes
                -----
                Inner closure used by BlackJAX NUTS when a custom likelihood
                is provided. Not part of public API.
                """
                lh_val = likelihood(x)
                prior = 0.5 * sum(jnp.sum(v**2) for v in x.values())
                return -lh_val - prior

        else:
            _logdensity_2arg = self._get_or_build_logdensity_fn()
            _da = self._data_args

            @jax.jit
            def logdensity_fn(x):
                """Evaluate log density with data_args bound from the enclosing scope.

                Parameters
                ----------
                x : dict
                    Parameter dict in unbounded space.

                Returns
                -------
                float
                    Log posterior (likelihood + priors).

                Notes
                -----
                Inner closure used by BlackJAX NUTS when no custom likelihood
                is provided. Captures ``_data_args`` from outer scope.
                Not part of public API.
                """
                return _logdensity_2arg(x, _da)

        warmup_key, sample_key = jax.random.split(key)
        n_warmup = min(200, n_samples)
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        (state, parameters), _ = warmup.run(warmup_key, pos_dict, num_steps=n_warmup)

        if verbose:
            logger.info("  Warmup done (%d steps). Sampling...", n_warmup)

        kernel = blackjax.nuts(logdensity_fn, **parameters).step

        @jax.jit
        def one_step(state, rng_key):
            """Execute one NUTS sampling step and return updated state.

            Parameters
            ----------
            state : blackjax.SamplerState
                Current MCMC sampler state.
            rng_key : jax.random.PRNGKey
                Random seed for this step.

            Returns
            -------
            tuple of (blackjax.SamplerState, blackjax.SamplerState)
                Updated state twice (for jax.lax.scan compatibility).

            Notes
            -----
            Inner closure for NUTS kernel. Designed for use with ``jax.lax.scan``
            in batch sampling. Not part of public API.
            """
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(sample_key, n_samples)
        _, states = jax.lax.scan(one_step, state, keys)

        sample_positions = states.position
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
            method == "map" and self.model._precomputed.photometry is not None and _same_shape
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
            and self.model._precomputed.photometry is not None
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
        import blackjax
        from jax.flatten_util import ravel_pytree

        from tengri.inference.backends.mcmc._shared import (
            _get_dynamic_hmc_kernel,
            _get_flat_logdensity,
            _get_ghmc_kernel,
            _get_hmc_kernel,
            _get_nuts_kernel,
        )
        from tengri.inference.posterior import Posterior

        n_warmup = kwargs.get("n_warmup", 300)
        n_burnin = kwargs.get("n_burnin", 100)
        n_samples = kwargs.get("n_samples", 1000)
        target_accept_rate = kwargs.get("target_accept_rate", 0.85)
        max_num_doublings = kwargs.get("max_num_doublings", 10)
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
        noise_inv_batch = 1.0 / noise_batch**2
        batch_data_args = {
            "data": flux_batch,
            "noise": noise_batch,
            "noise_inv": noise_inv_batch,
            "sqrt_noise_inv": jnp.sqrt(noise_inv_batch),
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

        # Run adaptation on first galaxy (shared across all)
        first_data_args = jax.tree.map(lambda x: x[0], batch_data_args)

        def ld_first(pos):
            """Log-density for the first galaxy (used in warmup adaptation).

            Parameters
            ----------
            pos : ndarray, shape (n_dim,)
                Flattened unbounded parameters for first galaxy.

            Returns
            -------
            float
                Log posterior density.

            Notes
            -----
            Inner closure for warmup adaptation in batch MCMC. Not part of
            public API.
            """
            return logdensity_flat_2arg(pos, first_data_args)

        if method in ("mcmc_nuts", "mcmc_hmc"):
            bj_algo = blackjax.nuts if method == "mcmc_nuts" else blackjax.hmc
            adapt_kwargs = {}
            if method == "mcmc_hmc":
                adapt_kwargs["num_integration_steps"] = n_leapfrog_steps
            warmup = blackjax.window_adaptation(
                bj_algo,
                ld_first,
                is_mass_matrix_diagonal=not use_dense,
                target_acceptance_rate=target_accept_rate,
                **adapt_kwargs,
            )
            (_, adapt_params), _ = warmup.run(adapt_key, init_flats[0], num_steps=n_warmup)
        elif method == "mcmc_dynamic_hmc":
            warmup = blackjax.window_adaptation(
                blackjax.hmc,
                ld_first,
                is_mass_matrix_diagonal=not use_dense,
                target_acceptance_rate=target_accept_rate,
                num_integration_steps=10,
            )
            (_, adapt_params), _ = warmup.run(adapt_key, init_flats[0], num_steps=n_warmup)
        elif method == "mcmc_ghmc":
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                ld_first,
                is_mass_matrix_diagonal=not use_dense,
                target_acceptance_rate=target_accept_rate,
            )
            (_, adapt_params), _ = warmup.run(adapt_key, init_flats[0], num_steps=n_warmup)

        step_size = adapt_params["step_size"]
        inv_mass_matrix = adapt_params["inverse_mass_matrix"]

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

            def _sample_scan(state, keys, data_args_i):
                """Scan over MCMC steps for a single galaxy (NUTS variant)."""

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

            def _sample_scan(state, keys, data_args_i):
                """Scan over MCMC steps for a single galaxy (HMC variant)."""

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

            def _sample_scan(state, keys, data_args_i):
                """Scan over MCMC steps for a single galaxy (dynamic HMC variant)."""

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
            if inv_mass_matrix.ndim == 2:
                momentum_inv_scale = jnp.sqrt(jnp.diag(inv_mass_matrix))
            else:
                momentum_inv_scale = jnp.sqrt(inv_mass_matrix)

            def _sample_scan(state, keys, data_args_i):
                """Scan over MCMC steps for a single galaxy (GHMC variant)."""

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
        def single_galaxy(gal_key, init_flat_i, data_args_i):
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
            ``kernel``, ``_sample_scan`` from outer scope. Not part of
            public API.
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
                state = blackjax.mcmc.ghmc.init(init_flat_i, init_key, ld)
            elif method == "mcmc_hmc":
                state = blackjax.mcmc.hmc.init(init_flat_i, ld)
            elif method == "mcmc_dynamic_hmc":
                state = blackjax.mcmc.dynamic_hmc.init(init_flat_i, ld, init_key)
            else:
                state = blackjax.mcmc.nuts.init(init_flat_i, ld)

            # Burn-in (discarded)
            if n_burnin > 0:
                burnin_keys = jax.random.split(burn_key, n_burnin)
                state, _ = _sample_scan(state, burnin_keys, data_args_i)

            # Sampling
            sample_keys = jax.random.split(sample_key, n_samples)
            _, (positions, divergent) = _sample_scan(state, sample_keys, data_args_i)
            return positions, divergent

        # vmap + jit: one XLA kernel for all galaxies
        gal_keys = jax.random.split(key, n_gal)
        all_positions, all_divergent = jax.jit(jax.vmap(single_galaxy))(
            gal_keys,
            init_flats,
            batch_data_args,
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
        noise_inv_batch = 1.0 / noise_batch**2
        batch_data_args = {
            "data": flux_batch,
            "noise": noise_batch,
            "noise_inv": noise_inv_batch,
            "sqrt_noise_inv": jnp.sqrt(noise_inv_batch),
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

            batch_result = jax.jit(jax.vmap(solver.run))(params_batch, batch_data_args)

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
                bounded_i = self._bounded_from_unbounded(params_i)
                result_i = Posterior(
                    samples=bounded_i,
                    log_weights=None,
                    fitter=self,
                    method=f"map ({opt_name})",
                    diagnostics={
                        "loss": float(batch_result.state.value[g_idx]),
                        "n_steps": int(batch_result.state.iter_num[g_idx]),
                        "optimizer": opt_name,
                        "converged": bool(batch_result.state.error[g_idx] < tol),
                    },
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

        batch_step = jax.jit(jax.vmap(single_step))

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
            bounded_i = self._bounded_from_unbounded(params_i)
            result_i = Posterior(
                samples=bounded_i,
                log_weights=None,
                fitter=self,
                method="map",
                diagnostics={"loss": float(losses[g_idx]), "n_steps": n_steps},
            )
            results.append(result_i)

        return results


# ── Backend Registry Initialization ──────────────────────────────────────────
# All ``@register_backend(...)`` calls live in ``inference/_registration.py``.
# That module is imported for its side effects by ``inference/__init__.py``,
# which guarantees the registry is populated before any caller can dispatch
# through ``Fitter.run``. See ADR-0010.
