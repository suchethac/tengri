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
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

__all__ = ["Fitter", "resolve_method"]

if TYPE_CHECKING:
    from tengri.inference.posterior import Posterior

import jax

logger = logging.getLogger(__name__)
import jax.numpy as jnp

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
from tengri.runtime.exceptions import ParameterError

# ── Method name unification ────────────────────────────────────────────

# Maps deprecated/old method strings → new canonical names.
_DEPRECATED_METHOD_ALIASES: dict[str, str] = {
    # Old nifty-qualified names → clean canonical
    "vi_nifty": "vi",
    "vi_nifty_linear": "vi_linear",
    # Old geoVI names → vi
    "geovi": "vi",
    "fast_geovi": "vi",
    "nifty_geovi": "vi",
    "geovi_nuts": "vi",
    # Old MGVI / linear names → vi_linear
    "mgvi": "vi_linear",
    "fast_mgvi": "vi_linear",
    "nifty_mgvi": "vi_linear",
    "evi": "vi_linear",
    # Old native names → native variants (were wrongly mapping to nifty)
    "native_geovi": "vi_native",
    "native_mgvi": "vi_native_linear",
    "native_evi": "vi_native_linear",
    # MCMC
    "raytrace": "mcmc_raytrace",
    "nuts": "mcmc_nuts",
    "hmc": "mcmc_hmc",
    "dynamic_hmc": "mcmc_dynamic_hmc",
    "ghmc": "mcmc_ghmc",
    "mclmc": "mcmc_mclmc",
    "adjusted_mclmc": "mcmc_adjusted_mclmc",
    "elliptical_slice": "mcmc_ess",
    # Evidence
    "evidence": "nss",
}

# D threshold for "auto": D <= this → mcmc_nuts, D > this → vi
_AUTO_D_THRESHOLD = 20

# D threshold for "mcmc": D <= this → NUTS, D > this → Ray Tracing
_MCMC_AUTO_D_THRESHOLD = 20

# Canonical method names (public API)
_CANONICAL_METHODS = {
    "vi",  # geoVI via NIFTy optimize_kl — default
    "vi_linear",  # MGVI via NIFTy optimize_kl
    "vi_nifty_fast",  # geoVI via NIFTy OptimizeVI.update (no logging)
    "vi_nifty_fast_linear",  # MGVI via NIFTy OptimizeVI.update
    "vi_native",  # Native JAX geoVI (experimental)
    "vi_native_linear",  # Native JAX MGVI (experimental)
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


def resolve_method(method: str, emit_warning: bool = True) -> str:
    """Resolve method string to canonical name.

    Maps deprecated aliases to their canonical names, emitting a
    DeprecationWarning if an alias is used. Validates that the final
    method is canonical or "auto".

    Parameters
    ----------
    method : str
        Method name (canonical, deprecated alias, "auto", or invalid).
    emit_warning : bool, optional
        If True (default), emit DeprecationWarning for deprecated aliases.

    Returns
    -------
    str
        Canonical method name.

    Raises
    ------
    ParameterError
        If method is not canonical, not a recognized alias, and not "auto".

    Examples
    --------
    >>> resolve_method("vi")
    'vi'
    >>> resolve_method("geovi")  # doctest: +SKIP
    'vi'  # and emits DeprecationWarning
    >>> resolve_method("vi_nifty")  # doctest: +SKIP
    'vi'  # and emits DeprecationWarning
    >>> resolve_method("invalid_method")
    ParameterError: Unknown method ...
    """
    if method is None:
        raise ParameterError(
            "method=None is not allowed. Pass an explicit method string "
            "(e.g. 'vi_nifty', 'mcmc_nuts', 'auto') or omit the argument to use "
            "the default from defaults.toml."
        )

    # If already canonical or "auto", return as-is
    if method in _CANONICAL_METHODS:
        return method

    # Check if deprecated alias
    if method in _DEPRECATED_METHOD_ALIASES:
        canonical = _DEPRECATED_METHOD_ALIASES[method]
        if emit_warning:
            warnings.warn(
                f"Method '{method}' is deprecated. Use '{canonical}' instead. "
                f"Old names will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=3,  # Caller's caller (skip resolve_method frame)
            )
        return canonical

    # Invalid method
    canonical_list = ", ".join(sorted(_CANONICAL_METHODS))
    raise ParameterError(
        f"Unknown method: '{method}'. "
        f"Valid canonical names: {canonical_list}. "
        f"Deprecated aliases: {', '.join(sorted(_DEPRECATED_METHOD_ALIASES.keys()))}. "
        f"See Fitter.run() docstring for details."
    )


class Fitter:
    """Inference engine for tengri models.

    Parameters
    ----------
    model : Model
        Configured forward model.
    data : array
        Observed data (photometry or spectrum).
    noise : array
        1-sigma uncertainties.
    data_type : str or None
        ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
        If ``None`` (default), inferred from ``model.observation``.
        Explicit values override the inferred type.
    calibration_marginalize : bool
        If ``True``, analytically marginalize over spectroscopic
        calibration polynomial coefficients (Chebyshev) when computing
        the spectroscopic log-likelihood.  Only applies when
        ``data_type`` is ``"spectroscopy"`` or ``"joint"``.
        Follows the Prospector approach (Johnson et al. 2021).
        Default ``False``.
    cal_n_poly : int
        Number of Chebyshev polynomial coefficients for calibration
        marginalization (order 1 through ``cal_n_poly``).  Default 3.
    cal_prior_sigma : float
        Standard deviation of the Gaussian prior on each calibration
        coefficient.  Default 1.0.
    eline_marginalize : bool or None
        Whether to analytically marginalize emission line amplitudes.
        ``None`` (default) auto-detects from the model's ``Spectroscopy``
        (uses ``eline_mode="marginalized"`` setting).
    eline_prior_type : str or None
        Prior type for emission lines: ``"flat"`` or ``"cloudy"``.
        ``None`` auto-detects from ``Spectroscopy.eline_prior_type``.
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(
        self,
        model,
        data,
        noise,
        data_type=None,
        data_mask=None,
        calibration_marginalize=False,
        cal_n_poly=3,
        cal_prior_sigma=1.0,
        eline_marginalize=None,
        eline_prior_type=None,
    ):
        # ── Data validation ─────────────────────────────────────────
        self.model = model
        self.data = jnp.asarray(data)
        self.noise = jnp.asarray(noise)
        self.data_mask = jnp.asarray(data_mask) if data_mask is not None else None
        self.data_type = self._resolve_data_type(data_type, model)
        self.spec = model.spec

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

        # ── Background compilation ─────────────────────────────────
        self._jit_sampler = None
        self._compilation_event = threading.Event()
        self._compilation_error: Exception | None = None
        self._compilation_lock = threading.Lock()
        self._compilation_thread: threading.Thread | None = None
        self._start_background_compilation()

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

        from tengri.components.sps.precompute import precompute_photometry

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

    def _start_background_compilation(self) -> None:
        """Spawn a daemon thread to pre-compile the JIT engine.

        XLA C++ compilation releases the GIL, so this runs in genuine
        parallel with the caller's Python setup code.  The
        ``_compilation_event`` is set before the first ``run("vi")``
        call can proceed past ``_get_or_build_engine``.

        Set ``TENGRI_NO_BACKGROUND_COMPILE=1`` to disable (used in tests
        to avoid spawning dozens of concurrent compilations per session).
        """
        import os

        if os.environ.get("TENGRI_NO_BACKGROUND_COMPILE"):
            # In test environments, skip background compilation entirely.
            # _get_or_build_engine will compile lazily on first run() call.
            self._compilation_event.set()
            return

        def _worker() -> None:
            try:
                with self._compilation_lock:
                    cache_key = self._engine_cache_key()
                    cache = get_model_cache(self.model).setdefault("jit_engine", {})
                    if cache_key not in cache:
                        self.compile(
                            modes=("linear_resample", "nonlinear_update"),
                            verbose=False,
                        )
            except Exception as exc:
                # Compilation errors (JAX tracing, type mismatches, etc.) are stored
                # and re-raised when the engine is first accessed via _get_or_build_engine()
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

        Engines are cached on the Model object so that multiple Fitters
        created with the same Model (but different data) share the same
        compiled XLA programs.

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

        cache_key = self._engine_cache_key()
        cache = get_model_cache(self.model).setdefault("jit_engine", {})
        if cache_key in cache:
            self._jit_sampler = cache[cache_key]
            return self._jit_sampler

        engine = self._build_jit_engine(pos_dict)
        cache[cache_key] = engine
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
            Which sample modes to pre-compile. Each mode compiles
            separately. Default covers MGVI + geoVI update (fastest).
            Add ``"nonlinear_resample"`` for full geoVI (~56s extra).
        verbose : bool
            Print compilation progress.

        Example
        -------
        >>> fitter = Fitter(model, data, noise)
        >>> fitter.compile()  # ~3s for default modes
        >>> fitter.compile(
        ...     modes=(  # ~60s for all modes
        ...         "linear_resample",
        ...         "nonlinear_update",
        ...         "nonlinear_resample",
        ...     )
        ... )
        >>> result = fitter.run("vi_native")  # instant
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

        if verbose:
            logger.info("Compilation complete.")
        return self

    # ── Loss and likelihood builders ──────────────────────────────────

    def _build_loss_fn(self, mode: str = "_traceable") -> Callable:
        """Build a differentiable loss function.

        See ``tengri.inference.loss_functions.build_loss_fn`` for full docs.
        Returns ``loss_fn(params_unbounded, data_args) -> scalar``.

        Parameters
        ----------
        mode : str, optional
            Forward model prediction mode. Default "_traceable" is for
            backward compatibility. Use "auto" for ~1.5x speedup with
            non-NIFTy methods.
        """
        return build_loss_fn(self, mode=mode)

    def _get_or_build_loss_fn(self, mode: str = "_traceable") -> Callable:
        """Return the cached loss function, building it if needed.

        The loss function is cached on the Model object keyed by
        ``_engine_cache_key()`` + mode so that multiple Fitters with the same
        model structure share the same compiled XLA program.

        Parameters
        ----------
        mode : str, optional
            Forward model prediction mode. Default "_traceable" for backward
            compatibility. Pass "auto" for ~1.5x speedup with non-NIFTy methods.
        """
        cache_key = (self._engine_cache_key(), mode)
        cache = get_model_cache(self.model).setdefault("loss_fn", {})
        if cache_key in cache:
            return cache[cache_key]
        loss_fn = self._build_loss_fn(mode=mode)
        cache[cache_key] = loss_fn
        return loss_fn

    def _build_logprior_fn(self) -> Callable:
        """Build a log-prior function. See ``loss_functions.build_logprior_fn``."""
        return build_logprior_fn(self)

    def _build_loglikelihood_fn(self, mode: str = "_traceable") -> Callable:
        """Build log-likelihood function. See ``loss_functions.build_loglikelihood_fn``."""
        return build_loglikelihood_fn(self, mode=mode)

    def _get_or_build_loglikelihood_fn(self, mode: str = "_traceable") -> Callable:
        """Return the cached log-likelihood function, building if needed."""
        cache_key = (self._engine_cache_key(), mode)
        cache = get_model_cache(self.model).setdefault("loglik_fn", {})
        if cache_key in cache:
            return cache[cache_key]
        loglik_fn = self._build_loglikelihood_fn(mode=mode)
        cache[cache_key] = loglik_fn
        return loglik_fn

    def _build_loglikelihood_unbounded_fn(self, mode: str = "_traceable") -> Callable:
        """Build unbounded-space log-likelihood.

        See ``loss_functions.build_loglikelihood_unbounded_fn``.
        """
        return build_loglikelihood_unbounded_fn(self, mode=mode)

    def _get_or_build_grad_fn(self, mode: str = "_traceable") -> Callable:
        """Return cached JIT-compiled value_and_grad of the loss function.

        The gradient function takes ``(params_unbounded, data_args)`` as
        explicit arguments so the compiled XLA program is reusable across
        galaxies with the same model structure.
        """
        cache_key = (self._engine_cache_key(), mode)
        cache = get_model_cache(self.model).setdefault("grad_fn", {})
        if cache_key in cache:
            return cache[cache_key]

        loss_fn = self._get_or_build_loss_fn(mode=mode)

        @jax.jit
        def val_and_grad(params_u, data_args):
            return jax.value_and_grad(lambda p: loss_fn(p, data_args))(params_u)

        cache[cache_key] = val_and_grad
        return val_and_grad

    def _get_or_build_logdensity_fn(self, mode: str = "_traceable") -> Callable:
        """Return cached JIT-compiled log-density for MCMC/Pathfinder.

        Returns ``logdensity(params_u, data_args) -> scalar``.  Callers
        should partial-apply ``data_args`` for blackjax compatibility.
        """
        cache_key = (self._engine_cache_key(), mode)
        cache = get_model_cache(self.model).setdefault("logdensity_fn", {})
        if cache_key in cache:
            return cache[cache_key]

        loss_fn = self._get_or_build_loss_fn(mode=mode)

        @jax.jit
        def logdensity(params_u, data_args):
            return -loss_fn(params_u, data_args)

        cache[cache_key] = logdensity
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

    def run(self, method: str = "vi", *, init_from=None, key=None, **kwargs):
        """Run inference.

        Parameters
        ----------
        method : str
            ``"vi"``                   — geoVI via NIFTy (nonlinear, default).
            ``"vi_linear"``            — MGVI via NIFTy (linearised).
            ``"vi_nifty_fast"``        — geoVI fast path (~35% faster, no logging).
            ``"vi_nifty_fast_linear"`` — MGVI fast path (~35% faster, no logging).
            ``"vi_native"``            — Native JAX geoVI (experimental).
            ``"vi_native_linear"``     — Native JAX MGVI (experimental).
            ``"mcmc_nuts"``            — NUTS via BlackJAX (default for D≤20).
            ``"mcmc_hmc"``             — Standard HMC (fixed trajectory length).
            ``"mcmc_dynamic_hmc"``     — Dynamic HMC (adaptive trajectory).
            ``"mcmc_ghmc"``            — Generalized HMC (partial momentum refresh).
            ``"mcmc_mclmc"``           — MCLMC (O(1) grad/sample, biased).
            ``"mcmc_adjusted_mclmc"``  — Adjusted MCLMC (Metropolis-corrected).
            ``"mcmc_raytrace"``        — Ray Tracing (Behroozi 2025).
            ``"mcmc"``                 — Auto: NUTS (D≤20) or Ray Tracing (D>20).
            ``"mcmc_ess"``             — Elliptical Slice Sampling.
            ``"map"``                  — MAP optimisation.
            ``"laplace"``              — Gaussian approximation at MAP.
            ``"pathfinder"``           — L-BFGS path (Zhang+2022).
            ``"nss"``                  — Nested Slice Sampling, log Z (D≤30).
            ``"auto"``                 — mcmc_nuts (D≤20) or vi (D>20).

            Deprecated aliases (still work, emit DeprecationWarning):
            ``"vi_nifty"``         → ``"vi"``
            ``"vi_nifty_linear"``  → ``"vi_linear"``
            ``"geovi"``, ``"fast_geovi"``,
            ``"nifty_geovi"``      → ``"vi"``
            ``"mgvi"``, ``"fast_mgvi"``,
            ``"nifty_mgvi"``       → ``"vi_linear"``
            ``"native_geovi"``     → ``"vi_native"``
            ``"native_mgvi"``, ``"native_evi"`` → ``"vi_native_linear"``
            ``"raytrace"``         → ``"mcmc_raytrace"``
            ``"nuts"``             → ``"mcmc_nuts"``
            ``"hmc"``              → ``"mcmc_hmc"``
            ``"dynamic_hmc"``      → ``"mcmc_dynamic_hmc"``
            ``"ghmc"``             → ``"mcmc_ghmc"``
            ``"mclmc"``            → ``"mcmc_mclmc"``
            ``"adjusted_mclmc"``   → ``"mcmc_adjusted_mclmc"``
            ``"elliptical_slice"`` → ``"mcmc_ess"``
            ``"evidence"``         → ``"nss"``

        init_from : Posterior, optional
            Use a previous result as warm-start initialisation.
        key : PRNGKey, optional
            Random key (defaults to PRNGKey(42)).
        **kwargs
            Method-specific arguments passed to the underlying sampler.

        Returns
        -------
        Posterior
            Inference results with ``._fitter`` back-reference set.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        # Resolve deprecated aliases and validate method
        method = resolve_method(method)

        # --- Merge TOML method-specific defaults (caller kwargs win) ---
        try:
            from tengri.parameters.defaults import get_inference_defaults

            kwargs = {**get_inference_defaults(method), **kwargs}
        except (ImportError, FileNotFoundError, OSError):
            # Config file unavailable or unreadable — skip defaults merge
            pass

        # Strip any stale vi_flavor kwarg that callers may pass (no longer used)
        kwargs.pop("vi_flavor", None)

        # --- "auto" method: dimensionality-based selection ---
        if method == "auto":
            d = self.spec.n_free
            try:
                from tengri.parameters.defaults import get_inference_defaults

                threshold = int(get_inference_defaults().get("mcmc_auto_d", _AUTO_D_THRESHOLD))
            except (ImportError, FileNotFoundError, OSError, KeyError, ValueError):
                # Config unavailable, missing key, or non-integer value — use hardcoded fallback
                threshold = _AUTO_D_THRESHOLD
            method = "mcmc_nuts" if d <= threshold else "vi"

        # --- Dispatch to underlying _run_* methods ---
        if method == "map":
            result = self._run_map(key=key, init_from=init_from, **kwargs)

        elif method == "vi":
            # geoVI via NIFTy optimize_kl (default)
            result = self._run_vi(key=key, init_from=init_from, **kwargs)

        elif method == "vi_linear":
            # MGVI via NIFTy optimize_kl
            result = self._run_vi_linear(key=key, init_from=init_from, **kwargs)

        elif method == "vi_nifty_fast":
            # geoVI fast path — NIFTy OptimizeVI.update, no logging
            result = self._run_nifty_fast_vi(key=key, init_from=init_from, **kwargs)

        elif method == "vi_nifty_fast_linear":
            # MGVI fast path — NIFTy OptimizeVI.update, no logging
            result = self._run_nifty_fast_vi_linear(key=key, init_from=init_from, **kwargs)

        elif method == "vi_native":
            # Native JAX geoVI — experimental, not production-ready
            result = self._run_vi_native(key=key, init_from=init_from, **kwargs)

        elif method == "vi_native_linear":
            # Native JAX MGVI — experimental, not production-ready
            result = self._run_vi_native_linear(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc":
            # Auto-select: NUTS for low-D (exact gold-standard), RT for high-D
            d = self.spec.n_free
            if d <= _MCMC_AUTO_D_THRESHOLD:
                result = self._run_nuts(key=key, init_from=init_from, **kwargs)
            else:
                result = self._run_raytrace(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_raytrace":
            result = self._run_raytrace(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_nuts":
            result = self._run_nuts(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_hmc":
            result = self._run_hmc(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_dynamic_hmc":
            result = self._run_dynamic_hmc(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_ghmc":
            result = self._run_ghmc(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_mclmc":
            result = self._run_mclmc(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_adjusted_mclmc":
            result = self._run_adjusted_mclmc(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_ess":
            result = self._run_elliptical_slice(key=key, init_from=init_from, **kwargs)

        elif method == "nss":
            result = self._run_nss(key=key, init_from=init_from, **kwargs)

        elif method == "laplace":
            result = self._run_laplace(key=key, init_from=init_from, **kwargs)

        elif method == "pathfinder":
            result = self._run_pathfinder(key=key, init_from=init_from, **kwargs)

        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Canonical names: 'vi', 'vi_linear', 'vi_nifty_fast', "
                f"'vi_nifty_fast_linear', 'vi_native', 'vi_native_linear', "
                f"'mcmc', 'mcmc_raytrace', 'mcmc_nuts', 'mcmc_hmc', "
                f"'mcmc_dynamic_hmc', 'mcmc_ghmc', 'mcmc_mclmc', "
                f"'mcmc_adjusted_mclmc', 'mcmc_ess', 'map', "
                f"'laplace', 'pathfinder', 'nss', 'auto'. "
                f"See Fitter.run() docstring for deprecated aliases."
            )

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
            "  Methods:     vi, vi_linear, vi_nifty_fast, vi_nifty_fast_linear, "
            "vi_native, vi_native_linear, mcmc, mcmc_raytrace, mcmc_nuts, "
            "mcmc_hmc, mcmc_dynamic_hmc, mcmc_ghmc, mcmc_mclmc, "
            "mcmc_adjusted_mclmc, mcmc_ess, map, laplace, pathfinder, nss, auto"
        )

        lines.append(sep)
        return "\n".join(lines)

    def _get_mode_for_method(self, method: str) -> str:
        """Determine forward model prediction mode based on inference method.

        PERFORMANCE NOTE (2026-04-18): Profiling shows mode="_traceable" is
        12.64x FASTER than mode="auto" (5.9ms vs 74.4ms) with stable timing.
        mode="auto" has pathological variance (std=504ms, 6.8x the mean) causing
        occasional 500ms+ outliers. Always use mode="_traceable" for inference.

        Parameters
        ----------
        method : str
            Inference method name (e.g., "vi", "mcmc_nuts", "map")

        Returns
        -------
        str
            Always returns "_traceable" for optimal performance across all
            inference methods. Previous "auto" mode had severe variance issues.

        See Also
        --------
        docs/dev/jit-optimization-report-2026-04-18.md : Full profiling analysis
        """
        # ALL methods now use _traceable for 12.64x speedup + stable timing
        # (mode="auto" variance pathology fixed 2026-04-18)
        return "_traceable"

    # ── Private method runners ────────────────────────────────────────

    def _run_vi(self, *, key, init_from=None, **kwargs) -> Posterior:
        from tengri.inference.backends.vi.nifty import run_nifty_vi

        kwargs.setdefault("sample_mode", "nonlinear_resample")
        return run_nifty_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_vi_linear(self, *, key, init_from=None, **kwargs) -> Posterior:
        from tengri.inference.backends.vi.nifty import run_nifty_vi

        kwargs.setdefault("sample_mode", "linear_resample")
        return run_nifty_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_vi_native(self, *, key, init_from=None, **kwargs) -> Posterior:
        from tengri.inference.backends.vi.native import run_native_vi

        kwargs.setdefault("sample_mode", "geovi")
        return run_native_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_vi_native_linear(self, *, key, init_from=None, **kwargs) -> Posterior:
        from tengri.inference.backends.vi.native import run_native_vi

        kwargs.setdefault("sample_mode", "linear")
        return run_native_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_nifty_fast_vi(self, *, key, init_from=None, **kwargs) -> Posterior:
        from tengri.inference.backends.vi.nifty import run_nifty_fast_vi

        kwargs.setdefault("sample_mode", "nonlinear_resample")
        return run_nifty_fast_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_nifty_fast_vi_linear(self, *, key, init_from=None, **kwargs) -> Posterior:
        from tengri.inference.backends.vi.nifty import run_nifty_fast_vi

        kwargs.setdefault("sample_mode", "linear_resample")
        return run_nifty_fast_vi(self, key=key, init_from=init_from, **kwargs)

    def _run_nss(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.evidence import run_nss

        return run_nss(self, key=key, **kwargs)

    def _run_map(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.map_dispatch import run_map

        return run_map(self, key=key, **kwargs)

    def _run_raytrace(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_raytrace

        return run_raytrace(self, key=key, **kwargs)

    def _run_nuts(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_nuts

        return run_nuts(self, key=key, **kwargs)

    def _run_hmc(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_hmc

        return run_hmc(self, key=key, **kwargs)

    def _run_dynamic_hmc(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_dynamic_hmc

        return run_dynamic_hmc(self, key=key, **kwargs)

    def _run_ghmc(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_ghmc

        return run_ghmc(self, key=key, **kwargs)

    def _run_mclmc(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_mclmc

        return run_mclmc(self, key=key, **kwargs)

    def _run_adjusted_mclmc(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_adjusted_mclmc

        return run_adjusted_mclmc(self, key=key, **kwargs)

    def _run_laplace(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.map_dispatch import run_laplace

        return run_laplace(self, key=key, **kwargs)

    def _run_pathfinder(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.map_dispatch import run_pathfinder

        return run_pathfinder(self, key=key, **kwargs)

    def _run_elliptical_slice(self, *, key, **kwargs) -> Posterior:
        from tengri.inference.backends.mcmc.common import run_elliptical_slice

        return run_elliptical_slice(self, key=key, **kwargs)

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
        verbose=True,
    ):
        """Draw posterior samples from the converged geoVI approximation.

        Parameters
        ----------
        method : str
            "jit" (default) — JIT-compiled CG solve, ~0.2ms/sample.
            "blackjax" — BlackJAX NUTS (independent MCMC, not geoVI).
            "nifty" — NIFTy draw_linear_residual (slow, ~540ms/sample).
        """
        if method == "jit":
            return self._draw_jit_samples(
                pos_dict, key, n_samples, existing_samples, verbose=verbose
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
                    pos_dict, key, n_samples, existing_samples, verbose=verbose
                )
        return self._draw_nifty_samples(
            likelihood, pos_dict, key, n_samples, existing_samples, verbose=verbose
        )

    def _draw_jit_samples(self, pos_dict, key, n_samples, existing_samples, *, verbose=True):
        """Draw geoVI linear residual samples via JIT-compiled CG.

        Same math as NIFTy's draw_linear_residual but fully JIT-compiled:
        1. Draw z = J^T sqrt(N^{-1}) eta1 + eta2  (eta_i ~ N(0,I))
        2. Solve M @ residual = z via CG  (M = J^T N^{-1} J + I)
        3. Sample = pos + residual

        ~2000x faster than NIFTy's Python-loop CG.
        """
        if verbose:
            logger.info("  Drawing %d posterior samples (JIT CG)...", n_samples)

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(pos_dict)

        engine = self._jit_sampler
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        pos_flat = flatten(pos_dict)
        draw_keys = jax.random.split(key, n_samples)
        residuals_flat = engine["draw_samples"](pos_flat, draw_keys, self._data_args)

        for i in range(n_samples):
            res = unflatten(residuals_flat[i])
            combined = {k: pos_dict[k] + res[k] for k in pos_dict}
            existing_samples.append(combined)

        return existing_samples

    def _draw_nonlinear_jit_samples(
        self, pos_dict, key, n_samples, existing_samples, *, verbose=True
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

        # Draw in batches to avoid OOM for large n_samples
        batch_size = min(n_samples, 50)
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
                lh_val = likelihood(x)
                prior = 0.5 * sum(jnp.sum(v**2) for v in x.values())
                return -lh_val - prior

        else:
            _logdensity_2arg = self._get_or_build_logdensity_fn()
            _da = self._data_args

            @jax.jit
            def logdensity_fn(x):
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
        verbose : bool
        **kwargs
            Passed to run() (n_iterations, n_samples, n_seeds, etc).

        Returns
        -------
        list of Posterior

        Example
        -------
        >>> galaxies = [{"flux_obs": f, "noise": n} for f, n in zip(fluxes, noises)]
        >>> results = fitter.fit_batch(galaxies)
        >>> # First: ~15s compile. Rest: ~2ms each (vi).
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

        from tengri.inference.backends.mcmc.common import (
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
                def ld(pos):
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
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
                def ld(pos):
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
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
                def ld(pos):
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
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
                def ld(pos):
                    return logdensity_flat_2arg(pos, data_args_i)

                def _step(s, k):
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
            def ld(pos):
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
