# SPDX-License-Identifier: BSD-3-Clause
"""JIT engine builder extracted from Fitter._build_jit_engine.

Module-level function that accepts a Fitter instance and a position dict,
returning the compiled-function dict used by all geoVI/MGVI/EVI inference
paths.  Extracted here to keep fitter.py under the 800-line project limit
and to make the JIT-compilation logic independently readable.

The geoVI path implements the same CG, Newton-CG, sample drawing, and
nonlinear curving algorithms as NIFTy. Mathematical equivalence with
``jft.optimize_kl`` is verified by the cross-validation tests.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp

from tengri.inference._model_cache import _default_owner as _model_cache_owner
from tengri.inference.likelihoods.gaussian import standardized_residual, whiten

__all__ = [
    "CompileCache",
    "build_jit_engine",
    "clear_shared_caches",
    "get_or_build_engine_cached",
    "get_or_build_signal_response",
    "is_lean_mode",
    "is_persistent_mode",
    "lean",
    "persistent",
]


@dataclass
class CompileCache:
    """Explicit owner of JIT compilation cache state.

    Manages a bounded LRU cache of compiled XLA executables, with tunable
    max size and mode (normal/lean/persistent). Replaces module-level globals
    for per-Fitter or per-CatalogFitter isolation.

    Parameters
    ----------
    max_entries : int, optional
        Maximum number of cache entries to hold before evicting oldest.
        Default 2; tune via TENGRI_ENGINE_CACHE_MAXSIZE env var or per-instance.
    mode : {'normal', 'lean', 'persistent'}, optional
        Cache behavior mode. Default 'normal'.

        - 'normal': keep all cached entries (existing behavior).
        - 'lean': call clear_shared_caches(..., scope='inference_body')
          before each inference to drop stale entries.
        - 'persistent': never clear automatically; user calls clear() manually.

    _store : OrderedDict
        Internal LRU dict. Do not access directly; use get_or_compile().
    _store_lock : threading.Lock
        Thread safety for concurrent Fitter instances.

    Attributes
    ----------
    max_entries : int
    mode : str
    """

    max_entries: int = field(
        default_factory=lambda: int(os.environ.get("TENGRI_ENGINE_CACHE_MAXSIZE", "2"))
    )
    mode: Literal["normal", "lean", "persistent"] = "normal"
    _store: OrderedDict[Any, Any] = field(default_factory=OrderedDict)
    _store_lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_compile(self, key: Any, build_fn):
        """Get cached value or build and store it.

        Parameters
        ----------
        key : hashable
            Cache key (typically a tuple of compile parameters).
        build_fn : callable
            Function to call if key is not in cache. Must return the value to cache.

        Returns
        -------
        Any
            Either the cached value or the result of build_fn().
        """
        with self._store_lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            value = build_fn()
            self._lru_insert(key, value)
            return value

    def _lru_insert(self, key: Any, value: Any) -> None:
        """Insert into LRU dict; evict oldest entries past max_entries."""
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._store_lock:
            self._store.clear()

    def set_mode(self, mode: Literal["normal", "lean", "persistent"]) -> None:
        """Set cache mode.

        Parameters
        ----------
        mode : {'normal', 'lean', 'persistent'}
            New mode.
        """
        if mode not in ("normal", "lean", "persistent"):
            raise ValueError(f"mode must be one of ('normal', 'lean', 'persistent'); got {mode!r}")
        self.mode = mode

    def memory_estimate_gb(self) -> float | None:
        """Estimate memory used by cached entries.

        Returns
        -------
        float or None
            Estimated total memory in GB, or None if the platform doesn't
            support XLA executable introspection (e.g., Metal on macOS).
            Only an order-of-magnitude estimate; individual executable sizes
            are platform-dependent.
        """
        try:
            # XLA executable size introspection (JAX >= 0.3.5).
            # This is a best-effort estimate and may not work on all platforms.
            # Currently, JAX does not expose XLA executable sizes through
            # public APIs on all platforms, so we return None as a placeholder.
            # Future: integrate with JAX's memory profiling tools when available.
            with self._store_lock:
                # Entries are typically dicts of compiled functions.
                # Direct size introspection not yet available on all platforms.
                pass
            return None
        except (AttributeError, TypeError):
            return None


# ── Module-level shared caches for cross-galaxy engine reuse ────────────────
# Each cached entry holds a compiled XLA executable that can be hundreds of MB.
# Without bounding, repeated fits with slightly different signatures (each
# notebook re-run, each parameter tweak) accumulate engines indefinitely and
# the process RSS grows by ~1 GB per fit. Hence: bounded LRU + explicit clear
# helper + an env-var opt-out.
#
# Tuning:
#   TENGRI_ENGINE_CACHE_MAXSIZE   default 2; max number of engines held.
#   TENGRI_DISABLE_SHARED_CACHES  if set to "1", caches are never populated
#                                 (every Fitter compiles fresh). Use this if
#                                 you can afford the recompile cost and want
#                                 hard guarantees on memory.
_ENGINE_CACHE_MAXSIZE = int(os.environ.get("TENGRI_ENGINE_CACHE_MAXSIZE", "2"))
_SHARED_CACHES_DISABLED = os.environ.get("TENGRI_DISABLE_SHARED_CACHES", "") == "1"

# Lean mode: when True, ``Fitter.run()`` calls ``clear_shared_caches()``
# before dispatching to the backend. This prevents within-run
# accumulation in single-fit notebooks (model build + ztable + mock +
# HMC + posterior-predictive each compiles GB-scale JIT graphs that
# don't share executable memory).  Off by default (preserves
# cross-galaxy reuse for PopulationFitter / CatalogFitter).  Toggle via
# the ``tengri.lean()`` context manager or env var.
_LEAN_MODE: bool = os.environ.get("TENGRI_LEAN", "") == "1"
_LEAN_MODE_LOCK = threading.Lock()

# Persistent mode: opt-out from the lean default. When True, ``Fitter.run()``
# skips the auto-clear entirely and keeps every prior compile in RAM.
# After smart lean (2026-05) was wired to keep the matching ``(sig, method)``
# entry across runs, ``persistent()`` is rarely needed — smart lean already
# preserves what catalog and same-method loops want. ``persistent()`` is now
# only useful for keeping *non-matching* entries (e.g. an MAP and HMC
# compile alive simultaneously, swapping back and forth).
_PERSISTENT_MODE: bool = os.environ.get("TENGRI_PERSISTENT", "") == "1"
_PERSISTENT_MODE_LOCK = threading.Lock()

# Module-level singleton CompileCache for backwards compatibility.
_SINGLETON_CACHE: CompileCache | None = None


def _get_singleton_cache() -> CompileCache:
    """Get or create the module-level singleton CompileCache."""
    global _SINGLETON_CACHE
    if _SINGLETON_CACHE is None:
        _SINGLETON_CACHE = CompileCache(max_entries=_ENGINE_CACHE_MAXSIZE, mode="normal")
    return _SINGLETON_CACHE


def is_lean_mode() -> bool:
    """Return True if explicit lean mode is currently active."""
    return _LEAN_MODE


def is_persistent_mode() -> bool:
    """Return True if persistent (cache-reuse) mode is currently active."""
    return _PERSISTENT_MODE


def persistent(enabled: bool = True):
    """Context manager: keep ALL cached compiled artefacts across ``Fitter.run()``.

    .. deprecated:: 2026-05
        Use ``CompileCache(mode='persistent')`` passed to ``Fitter(..., cache=...)``
        instead. The module-level context manager is maintained for backwards
        compatibility but will emit a DeprecationWarning.

    After smart lean (2026-05), this is rarely needed. Smart lean already
    keeps the entry that matches the upcoming run's
    ``(compile_signature, method)`` — so a CatalogFitter loop or repeated
    identical fit hits the cache without any wrapping context.

    Use ``persistent()`` only when you genuinely want to keep *non-matching*
    L3 entries alive — e.g. swapping back and forth between MAP and HMC
    on the same fitter and wanting both compiles in RAM simultaneously.

    Equivalent ``TENGRI_PERSISTENT=1`` env var sets it process-wide.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        global _PERSISTENT_MODE
        warnings.warn(
            "tengri.persistent() context manager is deprecated. "
            "Use CompileCache(mode='persistent') passed to Fitter(..., cache=...) instead. "
            "This deprecation shim will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        with _PERSISTENT_MODE_LOCK:
            prev = _PERSISTENT_MODE
            _PERSISTENT_MODE = bool(enabled)
        try:
            yield
        finally:
            with _PERSISTENT_MODE_LOCK:
                _PERSISTENT_MODE = prev

    return _ctx()


def lean(enabled: bool = True):
    """Context manager: clear inference-body caches before each ``Fitter.run()``.

    .. deprecated:: 2026-05
        Use ``CompileCache(mode='lean')`` passed to ``Fitter(..., cache=...)``
        instead. The module-level context manager is maintained for backwards
        compatibility but will emit a DeprecationWarning.

    Usage::

        with tengri.lean():
            fitter.run("map", ...)  # caches cleared (surgical)
            fitter.run("mcmc_hmc", ...)  # caches cleared (surgical)
            # posterior-predictive plotting

    Each ``run()`` call inside the block clears only the heavy inference
    scan body (``_SHARED_ENGINE_CACHE``), preserving shareable forward
    compiles (loss, gradient, logdensity, structural kernels). This keeps
    peak RSS bounded while avoiding recompile of the smaller forward graphs.

    Lean is now *surgical* — it drops the 5–6 GB inference loop body but
    reuses the 0.5–1.5 GB forward compiles across phases, a significant
    improvement over clearing everything (see Phase B in the compilation
    cache architecture).

    Safe inside ``CatalogFitter`` / ``PopulationFitter`` — smart lean
    keeps the matching entry across the loop, so you get reuse without
    needing ``persistent()``.

    Equivalent ``TENGRI_LEAN=1`` env var sets it process-wide.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        global _LEAN_MODE
        warnings.warn(
            "tengri.lean() context manager is deprecated. "
            "Use CompileCache(mode='lean') passed to Fitter(..., cache=...) instead. "
            "This deprecation shim will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        with _LEAN_MODE_LOCK:
            prev = _LEAN_MODE
            _LEAN_MODE = bool(enabled)
        try:
            yield
        finally:
            with _LEAN_MODE_LOCK:
                _LEAN_MODE = prev

    return _ctx()


_SHARED_ENGINE_CACHE: OrderedDict = OrderedDict()
_SHARED_ENGINE_CACHE_LOCK = threading.Lock()

_SHARED_SIGNAL_RESPONSE_CACHE: OrderedDict = OrderedDict()
_SHARED_SIGNAL_RESPONSE_CACHE_LOCK = threading.Lock()

# Loss / value-and-grad / log-density / log-likelihood caches.  Same
# key shape: (compile_signature, engine_cache_key, mode).  Enables reuse
# of the heaviest piece (the gradient compile) across:
#   - different inference backends on the same Fitter (MAP then HMC then VI),
#   - different Fitters on different galaxies that share shape signature.
# Each entry holds a JIT-compiled callable; cleared by clear_shared_caches.
_SHARED_LOSS_FN_CACHE: OrderedDict = OrderedDict()
_SHARED_LOSS_FN_CACHE_LOCK = threading.Lock()

_SHARED_GRAD_FN_CACHE: OrderedDict = OrderedDict()
_SHARED_GRAD_FN_CACHE_LOCK = threading.Lock()

_SHARED_LOGDENSITY_FN_CACHE: OrderedDict = OrderedDict()
_SHARED_LOGDENSITY_FN_CACHE_LOCK = threading.Lock()

_SHARED_LOGLIK_FN_CACHE: OrderedDict = OrderedDict()
_SHARED_LOGLIK_FN_CACHE_LOCK = threading.Lock()


_SHARED_CACHES: dict[str, tuple[OrderedDict, threading.Lock]] = {
    "loss": (_SHARED_LOSS_FN_CACHE, _SHARED_LOSS_FN_CACHE_LOCK),
    "grad": (_SHARED_GRAD_FN_CACHE, _SHARED_GRAD_FN_CACHE_LOCK),
    "logdensity": (_SHARED_LOGDENSITY_FN_CACHE, _SHARED_LOGDENSITY_FN_CACHE_LOCK),
    "loglik": (_SHARED_LOGLIK_FN_CACHE, _SHARED_LOGLIK_FN_CACHE_LOCK),
}


def get_or_build_cached(fitter, kind: str, builder):
    """Shared LRU cache for compiled `kind` functions.

    ``kind`` is one of ``"loss"``, ``"grad"``, ``"logdensity"``, ``"loglik"``.
    """
    cache, lock = _SHARED_CACHES[kind]
    if _SHARED_CACHES_DISABLED:
        return builder()
    key = fitter.compile_signature()
    with lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = builder()
        _lru_set(cache, key, value, _ENGINE_CACHE_MAXSIZE)
        return value


def _lru_set(cache: OrderedDict, key, value, maxsize: int) -> None:
    """Insert into an LRU dict; evict oldest entries past ``maxsize``."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > maxsize:
        cache.popitem(last=False)


def _key_matches_sig(key, keep_sig: tuple) -> bool:
    """True if ``key`` should be kept under ``keep_sig`` (prefix match).

    Cache keys are ``fitter.compile_signature()`` tuples. ``keep_sig``
    is the same shape and the match is a prefix-equality check, so
    callers may pass a shorter prefix to keep a broader bucket. Used
    by surgical lean to retain the entry that matches the call about
    to run while dropping stale entries from prior phases.
    """
    if not isinstance(key, tuple) or not isinstance(keep_sig, tuple):
        return key == keep_sig
    if len(key) < len(keep_sig):
        return False
    return key[: len(keep_sig)] == keep_sig


def clear_shared_caches(
    *,
    scope: str = "all",
    drop_xla: bool = True,
    keep_sig: tuple | None = None,
) -> None:
    """Drop cached compiled artefacts from tengri in this process.

    Parameters
    ----------
    scope : {"all", "inference_body"}, default "all"
        Scope of caches to clear:

        - ``"all"``: Clear *everything* — engines, loss fns, grad fns,
          logdensity fns, prediction kernels, per-model caches, and JAX's
          XLA cache (if ``drop_xla=True``). Used by ``tengri.gc()``.
        - ``"inference_body"``: Drop only the heavy inference-loop body
          (``_SHARED_ENGINE_CACHE``); preserve every forward-model
          cache (``_SHARED_SIGNAL_RESPONSE_CACHE``,
          ``_SHARED_LOSS_FN_CACHE``, ``_SHARED_GRAD_FN_CACHE``,
          ``_SHARED_LOGDENSITY_FN_CACHE``, ``_SHARED_LOGLIK_FN_CACHE``,
          structural kernels) and per-model caches.
          Suitable for ``lean()`` context manager — keeps shareable forward
          compiles across phases.

    drop_xla : bool, default True
        Also call ``jax.clear_caches()`` to release JAX's own XLA-executable
        and tracing caches. Set False if you have other live JAX programs
        in the same process that you want to keep compiled.

    keep_sig : tuple, optional
        If supplied, entries whose key starts with this signature are
        preserved while every other entry in the affected caches is
        dropped. Used by ``Fitter.run(lean=True)`` to drop *stale* prior-
        phase compiles while keeping the entry that matches the current
        fitter — so a CatalogFitter loop over N galaxies all of the same
        shape pays exactly one compile, not N. ``None`` (the default)
        clears every entry in scope.

    Notes
    -----
    Recommended pattern for notebooks that run multiple inference phases
    in the same kernel (model build → MAP → HMC → posterior-predictive)::

        import tengri

        # ... model build, ztable precompute, run MAP ...
        result_map = fitter.run("map", ...)

        # ... HMC (with surgical lean, not full clear) ...
        post = fitter.run("mcmc_hmc", ...)

        # ... posterior-predictive plotting ...

    With the default ``lean=True`` in ``Fitter.run()``, the context
    automatically uses ``scope="inference_body"``, keeping forward compiles
    while dropping the 5–6 GB inference scan body.

    What gets cleared by scope="all"
    --------------------------------

    - All caches listed under scope="inference_body"
    - Cross-fitter structural kernel cache (prediction kernels)
    - JAX's process-internal caches (when ``drop_xla=True``)

    What does *not* get cleared
    ---------------------------

    - The on-disk persistent JAX compile cache (``~/.cache/tengri_jax_cache``).
      Use ``tengri.clear_cache()`` for that.

    """
    if scope not in ("all", "inference_body"):
        raise ValueError(f"scope must be 'all' or 'inference_body', got {scope!r}")

    def _drop(cache: OrderedDict, lock: threading.Lock) -> None:
        with lock:
            if keep_sig is None:
                cache.clear()
            else:
                stale = [k for k in cache if not _key_matches_sig(k, keep_sig)]
                for k in stale:
                    cache.pop(k, None)

    # Engine cache is the heavy inference-loop artefact (5-6 GB).
    # Signal-response is a forward-model JIT keyed on _engine_cache_key()
    # (a different shape from compile_signature) — keep_sig prefix
    # matching cannot preserve it, so clearing it on every Fitter.run
    # would silently nuke a useful forward compile. Treat it like the
    # other forward-model caches (loss/grad/logdensity) and only drop
    # it at scope="all".
    _drop(_SHARED_ENGINE_CACHE, _SHARED_ENGINE_CACHE_LOCK)

    # For scope="all", also drop structural kernels and all shared function caches
    if scope == "all":
        with _SHARED_SIGNAL_RESPONSE_CACHE_LOCK:
            _SHARED_SIGNAL_RESPONSE_CACHE.clear()
        with _SHARED_LOSS_FN_CACHE_LOCK:
            _SHARED_LOSS_FN_CACHE.clear()
        with _SHARED_GRAD_FN_CACHE_LOCK:
            _SHARED_GRAD_FN_CACHE.clear()
        with _SHARED_LOGDENSITY_FN_CACHE_LOCK:
            _SHARED_LOGDENSITY_FN_CACHE.clear()
        with _SHARED_LOGLIK_FN_CACHE_LOCK:
            _SHARED_LOGLIK_FN_CACHE.clear()

        from tengri.inference._sample_utils import _clear_vmap_to_physical_cache

        # Drops both the structural prediction kernels and the per-model
        # namespaces (loss/grad/engine handles). scope="inference_body"
        # deliberately leaves per-model caches in place — see the scope
        # contract in the docstring above.
        _model_cache_owner.clear()
        _clear_vmap_to_physical_cache()

    if drop_xla:
        import contextlib

        with contextlib.suppress(AttributeError):
            jax.clear_caches()

    import gc

    gc.collect()


def get_or_build_engine_cached(fitter, pos_dict):
    """Get or build a JIT engine from the shared cross-galaxy cache.

    Two Fitters with the same compile_signature() will receive the same
    compiled engine, enabling zero-recompile fits across different SEDModel
    instances that share shape and structure (e.g., different SSP files of
    the same shape in a catalog fit).

    Parameters
    ----------
    fitter : Fitter
        The Fitter instance requesting the engine.
    pos_dict : dict
        Position dict for computing static shapes.

    Returns
    -------
    dict
        Compiled engine with functions for all inference methods.
    """
    if _SHARED_CACHES_DISABLED:
        return build_jit_engine(fitter, pos_dict)

    sig = fitter.compile_signature()

    with _SHARED_ENGINE_CACHE_LOCK:
        if sig in _SHARED_ENGINE_CACHE:
            _SHARED_ENGINE_CACHE.move_to_end(sig)
            return _SHARED_ENGINE_CACHE[sig]

        engine = build_jit_engine(fitter, pos_dict)
        _lru_set(_SHARED_ENGINE_CACHE, sig, engine, _ENGINE_CACHE_MAXSIZE)
        return engine


def _build_signal_response(fitter):
    """Build data-free ``signal_response`` and ``_primals_to_params`` closures.

    Returns the *uncompiled* closures.  The caller decides whether to wrap
    with ``jax.jit`` (for standalone eager use) or leave unwrapped (so an
    outer ``jax.jit`` can inline it and differentiate through it).

    Neither closure captures any galaxy data — they depend only on model
    structure and parameter configuration.  This makes them the
    "reproducible components" that can be compiled once and shared across
    all Fitters for the same model.
    """
    model = fitter.model
    data_type = fitter.data_type
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    stochastic = spec.stochastic
    use_components = bool(getattr(fitter, "use_components", False))

    def _primals_to_params(primals):
        """Convert unbounded (standardized) primals dict to bounded physical params."""
        params = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            params[name] = dist.unstandardize(primals[name])
        for name, val in fixed_values.items():
            params[name] = val
        if stochastic and "psd_xi" in primals:
            params["psd_xi"] = primals["psd_xi"]
        params = spec.resolve_mirrors(params)
        return params

    def signal_response(primals):
        """Compute predicted data from unbounded parameters."""
        params = _primals_to_params(primals)
        if data_type == "photometry":
            if use_components:
                return model._photometry_via_state(params)
            return model.predict_photometry(params)
        elif data_type == "spectroscopy":
            if use_components:
                return model._spectrum_via_state(params)
            return model.predict_spectrum(params)
        elif data_type == "joint":
            if use_components:
                p = model._photometry_via_state(params)
                s = model._spectrum_via_state(params)
            else:
                p = model.predict_photometry(params)
                s = model.predict_spectrum(params)
            return jnp.concatenate([p, s])
        raise ValueError(f"Unknown data_type: {data_type}")

    return signal_response, _primals_to_params


def get_or_build_signal_response(fitter):
    """Return ``(signal_response, signal_response_jit)`` from shared cache.

    The physics stack (stellar populations, dust, nebular, AGN) is the
    "reproducible component": it does not depend on any galaxy's data.
    Caching it in a module-level dict enables cross-galaxy reuse:

    1. **Native path** — the same ``signal_response`` closure is used inside
       ``build_jit_engine``.  Because the outer ``run_evi_geovi_jit`` is
       itself cached per compile_signature(), the physics is traced only
       once per model structure regardless of galaxy count.

    2. **NIFTy path** — ``signal_response_jit`` is the stable function object
       passed to ``jft.Model``.  JAX's trace cache is keyed by Python function
       identity, so both ``run_nifty_vi`` (full path) and ``run_nifty_fast_vi``
       (tight loop) share one compiled physics kernel.  NIFTy's
       ``OptimizeVI.update`` skips re-tracing the SPS/dust/AGN stack for
       every new galaxy.

    Keyed by ``_engine_cache_key()`` (captures data_type, param names, model
    structure — but not galaxy data values or shapes).
    """
    from tengri.utils.compile_log import instrument_first_call

    cache_key = fitter._engine_cache_key()

    if _SHARED_CACHES_DISABLED:
        signal_response, _ = _build_signal_response(fitter)
        signal_response_jit = instrument_first_call(
            jax.jit(signal_response),
            "signal_response",
            fitter.compile_signature(),
        )
        return (signal_response, signal_response_jit)

    with _SHARED_SIGNAL_RESPONSE_CACHE_LOCK:
        if cache_key in _SHARED_SIGNAL_RESPONSE_CACHE:
            _SHARED_SIGNAL_RESPONSE_CACHE.move_to_end(cache_key)
            result = _SHARED_SIGNAL_RESPONSE_CACHE[cache_key]
        else:
            signal_response, _ = _build_signal_response(fitter)
            signal_response_jit = instrument_first_call(
                jax.jit(signal_response),
                "signal_response",
                fitter.compile_signature(),
            )
            result = (signal_response, signal_response_jit)
            _lru_set(
                _SHARED_SIGNAL_RESPONSE_CACHE,
                cache_key,
                result,
                _ENGINE_CACHE_MAXSIZE,
            )

    per_model_cache = _model_cache_owner.get_or_compile_model(fitter.model).setdefault(
        "signal_response", {}
    )
    per_model_cache[cache_key] = result
    return result


def build_jit_engine(fitter, pos_dict):
    """Build JIT-compiled inference engine: optimizer + posterior sampler.

    Returns a dict with compiled functions for the full EVI pipeline.
    All functions operate on flat arrays and use jax.lax.while_loop
    for zero Python overhead.

    The geoVI path uses NIFTy's actual implementations of CG,
    Newton-CG, sample drawing, and nonlinear curving — imported
    directly and called within the JIT boundary. This ensures
    mathematical equivalence with ``jft.optimize_kl``.

    Parameters
    ----------
    fitter : Fitter
        Configured Fitter instance (read-only; only attributes are accessed).
    pos_dict : dict
        Position dict mapping parameter names to initial JAX arrays.
        Used only to compute static shapes for flatten/unflatten.

    Returns
    -------
    dict
        Compiled functions: run_evi, run_evi_geovi, nifty_model,
        draw_samples, draw_nonlinear_samples, flatten, unflatten, etc.
    """
    from tengri.observation.noise import (
        compute_std_inv,
        get_noise_dof,
        has_noise_model,
        uses_student_t,
        variable_noise_hamiltonian,
        variable_noise_metric_vec,
    )

    # Import NIFTy for the exact geoVI path
    try:
        import nifty8.re  # noqa: F401

        _has_nifty = True
    except ImportError:
        _has_nifty = False

    model = fitter.model
    data_type = fitter.data_type
    # data/noise are NO LONGER captured here as local variables.
    # Instead they are passed at call-time via the ``data_args`` dict
    # so that the compiled engine can be reused across galaxies.
    use_variable_noise = has_noise_model(fitter.spec)
    noise_dof = get_noise_dof(fitter.spec) if uses_student_t(fitter.spec) else None
    use_components = bool(getattr(fitter, "use_components", False))

    # --- Signal response (physics only) ---
    # NOT JIT'd here — must remain traceable so jax.jvp/vjp (in metric_vec)
    # and jax.value_and_grad (in hamiltonian) can differentiate through it.
    # The JIT-compiled version lives in get_or_build_signal_response() for
    # NIFTy and eager use; this unwrapped copy is for the native VI path.
    signal_response, _primals_to_params = _build_signal_response(fitter)

    # memory_mode="low": rematerialize the forward model on the backward
    # pass instead of caching activations. Cuts the autodiff-tape peak
    # for `metric_vec` (jvp+vjp) and `hamiltonian` (value_and_grad) at
    # the cost of recomputing the forward once per gradient evaluation.
    # Targets the high-D stochastic-SFH case where the tape blows up.
    if getattr(fitter, "_memory_mode", "fast") == "low":
        signal_response = jax.checkpoint(signal_response)

    # --- Signal + noise response for variable noise ---
    if use_variable_noise:

        def signal_noise_response(primals, data_args):
            """Return (predicted, std_inv) tuple for variable noise metric."""
            params = _primals_to_params(primals)
            if data_type == "photometry":
                if use_components:
                    predicted = model._photometry_via_state(params)
                else:
                    predicted = model.predict_photometry(params)
            elif data_type == "spectroscopy":
                if use_components:
                    predicted = model._spectrum_via_state(params)
                else:
                    predicted = model.predict_spectrum(params)
            elif data_type == "joint":
                if use_components:
                    p = model._photometry_via_state(params)
                    s = model._spectrum_via_state(params)
                else:
                    p = model.predict_photometry(params)
                    s = model.predict_spectrum(params)
                predicted = jnp.concatenate([p, s])
            else:
                raise ValueError(f"Unknown data_type: {data_type}")
            f_cal = params.get("noise_frac_cal", 0.0)
            noise = data_args["noise"]
            std_inv = compute_std_inv(noise, predicted, f_cal)
            return predicted, std_inv

    # --- Flatten/unflatten (static shapes) ---
    param_keys = sorted(pos_dict.keys())
    slices = []
    idx = 0
    for k in param_keys:
        arr = jnp.atleast_1d(pos_dict[k]).ravel()
        shape = jnp.atleast_1d(pos_dict[k]).shape
        slices.append((idx, idx + arr.shape[0], shape))
        idx += arr.shape[0]
    d_total = idx

    def flatten(d):
        """Flatten parameter dict to 1D vector."""
        return jnp.concatenate([jnp.atleast_1d(d[k]).ravel() for k in param_keys])

    def unflatten(x):
        """Unflatten 1D vector to parameter dict."""
        d = {}
        for i_k, k in enumerate(param_keys):
            start, end, shape = slices[i_k]
            val = jax.lax.dynamic_slice(x, (start,), (end - start,)).reshape(shape)
            if shape == (1,):
                val = val[0]
            d[k] = val
        return d

    # --- Core primitives ---
    _eps = 6.0 * jnp.finfo(jnp.float64).eps

    if use_variable_noise:

        def metric_vec(xi, v, data_args):
            """GGN metric for VariableCovarianceGaussian likelihood."""
            data = data_args["data"]

            def _snr(primals):
                """Compute signal and noise response for variable-noise likelihood."""
                return signal_noise_response(primals, data_args)

            return variable_noise_metric_vec(xi, v, _snr, data, unflatten, flatten)

        def hamiltonian(xi, data_args):
            """E_lh + 0.5 ||xi||^2 with variable noise (includes logdet)."""
            data = data_args["data"]
            noise = data_args["noise"]
            primals = unflatten(xi)
            params = _primals_to_params(primals)
            pred = signal_response(primals)
            f_cal = params.get("noise_frac_cal", 0.0)
            return variable_noise_hamiltonian(
                data, noise, pred, f_cal, dof=noise_dof
            ) + 0.5 * jnp.sum(xi**2)

    else:

        def metric_vec(xi, v, data_args):
            r"""M(xi) @ v = (J/sigma)^T (J/sigma) v + v.

            Algebraically ``J^T N^{-1} J v + v``, but never forming
            :math:`N^{-1} = 1/\sigma^2` — that is ~1e59 at a real photometric
            sigma and simply does not exist in float32, so the shipped
            spelling returned ``inf``, or ``NaN`` wherever ``Jv`` was exactly
            zero. Whitening twice keeps every intermediate representable
            (#1206). Same measure as ``diagnostics/fisher.py`` (#1542).
            """
            noise = data_args["noise"]
            xi_d, v_d = unflatten(xi), unflatten(v)
            _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
            _, vjp_fn = jax.vjp(signal_response, xi_d)
            return flatten(vjp_fn(whiten(whiten(Jv, noise), noise))[0]) + v

        def hamiltonian(xi, data_args):
            """H(xi) = 0.5 chi2 + 0.5 ||xi||^2."""
            data = data_args["data"]
            noise = data_args["noise"]
            pred = signal_response(unflatten(xi))
            chi2 = jnp.sum(standardized_residual(data, pred, noise) ** 2)
            return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

    def H_vg(xi, data_args):
        """Hamiltonian value and gradient w.r.t. xi only."""
        return jax.value_and_grad(lambda x: hamiltonian(x, data_args))(xi)

    _tiny = 6.0 * jnp.finfo(jnp.float64).tiny
    _n_reset = 20

    def cg_solve(mat_fn, b, x0, maxiter=30, miniter=6, absdelta=0.0, resnorm=0.0):
        """CG solve: mat_fn(x) = b.

        Implements NIFTy's ``_static_cg`` algorithm exactly (conjugate_gradient.py:217-388)
        for flat arrays.  Residual-norm (L2) is the primary convergence
        criterion; energy-based absdelta is secondary.  Negative curvature
        on the first CG iteration triggers a steepest-descent fallback.
        """
        r = mat_fn(x0) - b
        d = r
        gamma = jnp.dot(r, r)
        energy = jnp.dot((r - b) / 2, x0)
        init_info = jnp.where(gamma == 0.0, jnp.int32(0), jnp.int32(-2))
        init = (x0, r, d, gamma, energy, init_info, jnp.int32(0))

        def cond(s):
            """Continue CG loop if not converged."""
            return s[5] < -1

        def body(s):
            """Execute one CG iteration."""
            pos, r, d, prev_gamma, prev_energy, info, i = s
            i = i + 1

            q = mat_fn(d)
            curv = jnp.dot(d, q)
            alpha = prev_gamma / curv

            # Negative / zero curvature (NIFTy cg:278-286)
            info = jnp.where(curv <= 0.0, jnp.int32(0), info)
            alpha = jnp.where(curv <= 0.0, 0.0, alpha)
            pos = pos - alpha * d
            # First iter + negative curvature: steepest-descent fallback
            pos = jnp.where(
                (curv < 0.0) & (i <= 1),
                prev_energy / (-curv) * (-b),
                pos,
            )

            # Periodic residual reset (NIFTy cg:287-291)
            r_reset = mat_fn(pos) - b
            r_step = r - q * alpha
            r = jnp.where((i % _n_reset == 0) & (info < -1), r_reset, r_step)

            gamma = jnp.dot(r, r)

            # Tiny gamma (NIFTy cg:295)
            info = jnp.where(
                (gamma >= 0.0) & (gamma <= _tiny) & (info != -1),
                jnp.int32(0),
                info,
            )

            # Residual norm -- PRIMARY (NIFTy cg:296-298, norm_ord=2)
            r_norm = jnp.sqrt(gamma)
            info = jnp.where(
                (resnorm > 0.0) & (r_norm < resnorm) & (i >= miniter) & (info != -1),
                jnp.int32(0),
                info,
            )

            # Energy -- SECONDARY (NIFTy cg:301-313)
            energy = jnp.dot((r - b) / 2, pos)
            energy_diff = prev_energy - energy
            neg_energy_eps = -_eps * jnp.abs(energy)
            info = jnp.where(
                energy_diff < neg_energy_eps,
                jnp.where(info < -1, i, info),
                info,
            )
            info = jnp.where(
                (absdelta > 0.0) & (energy_diff < absdelta) & (i >= miniter) & (info != -1),
                jnp.int32(0),
                info,
            )

            # Maxiter (NIFTy cg:314)
            info = jnp.where((i >= maxiter) & (info != -1), i, info)

            # Update search direction (NIFTy cg:316)
            d = d * jnp.maximum(0.0, gamma / prev_gamma) + r

            return (pos, r, d, gamma, energy, info, i)

        return jax.lax.while_loop(cond, body, init)[0]

    # --- Posterior sampler: draw linear residuals ---
    def draw_residuals(pos_f, subkeys, data_args):
        """Draw n linear residual samples (vmapped)."""
        sqrt_ni = data_args["sqrt_noise_inv"]

        def draw_one(subkey):
            """Draw one posterior residual sample."""
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=(d_total,))
            # Whitened-data residual draw matches the data array's shape — 1-D
            # ``(n_pix,)`` for a single galaxy, ``(N_gal, n_pix)`` for a
            # hierarchical fit. Deriving it from ``sqrt_ni`` keeps the engine
            # topology-agnostic (was ``shape=(len(fitter.data),)``, which
            # collapsed a batched ``(N_gal, n_pix)`` data array to ``N_gal``
            # and broke population spectroscopy — suchethac/tengri#711).
            eta_lh = jax.random.normal(k2, shape=sqrt_ni.shape)
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
            return cg_solve(
                lambda v: metric_vec(pos_f, v, data_args),
                jt + eta_pr,
                eta_pr,
                maxiter=30,
                miniter=6,
                absdelta=1e-4,
            )

        return jax.vmap(draw_one)(subkeys)

    def _draw_batch_fn(pos_f, k, data_args):
        """Batch draw residuals for a single galaxy (before vmap)."""
        return draw_residuals(pos_f, k, data_args)

    draw_batch = jax.jit(jax.vmap(_draw_batch_fn, in_axes=(None, 0, None)))

    # --- geoVI: nonlinear coordinate transform primitives ---

    def transformation_flat(pos_f, data_args):
        """t(x) = sqrt(N^{-1}) @ f(x). Maps to whitened data-space."""
        sqrt_ni = data_args["sqrt_noise_inv"]
        return sqrt_ni * signal_response(unflatten(pos_f))

    def left_sqrt_metric_flat(pos_f, v_data, data_args):
        """L^T(pos) @ v = J^T(pos) @ sqrt(N^{-1}) @ v.

        Maps whitened data-space vector to parameter-space.
        Matches NIFTy's ``likelihood.left_sqrt_metric(pos, v)``
        for the Gaussian case.
        """
        sqrt_ni = data_args["sqrt_noise_inv"]
        _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
        return flatten(vjp_fn(sqrt_ni * v_data)[0])

    def right_sqrt_metric_flat(pos_f, v_param, data_args):
        """L(pos) @ v = sqrt(N^{-1}) @ J(pos) @ v.

        Maps parameter-space vector to whitened data-space.
        Matches NIFTy's ``likelihood.right_sqrt_metric(pos, v)``
        for the Gaussian case.
        """
        sqrt_ni = data_args["sqrt_noise_inv"]
        _, Jv = jax.jvp(signal_response, (unflatten(pos_f),), (unflatten(v_param),))
        return sqrt_ni * Jv

    def draw_metric_sample(pos_f, subkey, data_args):
        """Draw one sample with covariance M = J^T N^{-1} J + I.

        This is ``draw_linear_residual(..., from_inverse=False)``
        in NIFTy. The metric sample is NOT CG-inverted.
        """
        sqrt_ni = data_args["sqrt_noise_inv"]
        k1, k2 = jax.random.split(subkey)
        eta_pr = jax.random.normal(k1, shape=(d_total,))
        # See ``draw_residuals``: shape follows the data array, not its len().
        eta_lh = jax.random.normal(k2, shape=sqrt_ni.shape)
        _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
        jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
        return jt + eta_pr

    def _newton_cg_flat(
        fun_and_grad,
        hessp,
        x0,
        custom_gradnorm=None,
        maxiter=10,
        miniter=0,
        xtol=1e-3,
        energy_reduction_factor=0.1,
    ):
        """Newton-CG with successive-halving line search.

        Implements NIFTy's ``_static_newton_cg`` algorithm exactly (optimize.py:285-449)
        for flat arrays.  Includes adaptive CG tolerance, steepest-descent
        reset after 5 line-search halvings, and custom gradient norm.
        """
        ncg_xtol = xtol * d_total  # NIFTy: xtol * size(x0)

        def gradnorm(v):
            """Compute norm of gradient vector (L1 by default)."""
            if custom_gradnorm is not None:
                return custom_gradnorm(v)
            return jnp.sum(jnp.abs(v))  # L1 norm (NIFTy default)

        energy, g = fun_and_grad(x0)
        init_state = (
            x0,
            energy,
            jnp.array(jnp.inf),
            g,
            jnp.where(maxiter == 0, jnp.int32(0), jnp.int32(-2)),
            jnp.int32(0),
        )

        def ncg_cond(state):
            """Continue Newton-CG if not converged."""
            return state[4] < -1

        def ncg_body(state):
            """Execute one Newton-CG iteration with line search."""
            pos, energy, old_energy, g, status, i = state
            i = i + 1

            # Adaptive CG tolerance (NIFTy optimize.py:351-358)
            cg_abd_fallback = jnp.array(0.0, dtype=energy.dtype)
            cg_absdelta = jnp.where(
                ~jnp.isinf(old_energy),
                energy_reduction_factor * (old_energy - energy),
                cg_abd_fallback,
            )
            cg_absdelta = jnp.array(cg_absdelta, dtype=energy.dtype)

            # CG resnorm (NIFTy optimize.py:359-360, norm_ord=1)
            mag_g = jnp.sum(jnp.abs(g))
            cg_resnorm = jnp.minimum(0.5, jnp.sqrt(mag_g)) * mag_g

            # CG solve (NIFTy: norm_ord=1, _raise_nonposdef=False)
            nat_g = cg_solve(
                lambda v: hessp(pos, v),
                g,
                jnp.zeros_like(pos),
                maxiter=min(200, 20 * d_total),
                miniter=min(6, min(200, 20 * d_total)),
                absdelta=cg_absdelta,
                resnorm=cg_resnorm,
            )

            # Line search: successive halving (NIFTy optimize.py:452-523)
            # State: (status, iter, new_pos, new_energy, new_g,
            #         dd, grad_scaling, reset, nhev)
            ls_init = (
                jnp.int32(-2),
                jnp.int32(0),
                pos,
                jnp.array(jnp.inf),
                g,
                nat_g,
                1.0,
                jnp.bool_(False),
                jnp.int32(0),
            )

            def ls_cond(ls):
                """Continue line search if not successful."""
                return ls[0] < -1

            def ls_body(ls):
                """Execute one line search step (successive halving)."""
                (
                    ls_st,
                    ls_i,
                    _np,
                    _ne,
                    _ng,
                    dd,
                    gs,
                    reset,
                    nhev,
                ) = ls
                new_pos = pos - gs * dd
                new_e, new_g = fun_and_grad(new_pos)
                ls_st = jnp.where(new_e <= energy, jnp.int32(0), ls_st)
                gs = jnp.where(ls_st < -1, gs / 2.0, gs)
                # Steepest descent reset at iteration 5
                do_reset = (ls_i == 5) & (ls_st < -1)
                reset = jnp.where(do_reset, jnp.bool_(True), reset)
                gs = jnp.where(do_reset, 1.0, gs)
                gam = jnp.dot(g, g)
                curv = jnp.dot(g, hessp(pos, g))
                sd_dd = gam / curv * g
                dd = jnp.where(do_reset, sd_dd, dd)
                nhev = nhev + do_reset.astype(jnp.int32)
                # Abort after 8 iterations
                do_abort = (ls_i == 8) & (ls_st < -1)
                ls_st = jnp.where(do_abort, jnp.int32(-1), ls_st)
                return (
                    ls_st,
                    ls_i + 1,
                    new_pos,
                    new_e,
                    new_g,
                    dd,
                    gs,
                    reset,
                    nhev,
                )

            ls_result = jax.lax.while_loop(ls_cond, ls_body, ls_init)
            (
                ls_status,
                ls_iter,
                new_pos,
                new_energy,
                new_g,
                dd,
                gs,
                _reset,
                _nhev,
            ) = ls_result

            status = jnp.where(ls_status != 0, jnp.int32(-1), status)

            # Update only if line search succeeded (NIFTy opt:381-385)
            success = status < -1
            old_energy = jnp.where(success, energy, old_energy)
            energy_out = jnp.where(success, new_energy, energy)
            energy_diff = jnp.where(success, old_energy - energy_out, 0.0)
            pos_out = jnp.where(success, new_pos, pos)
            g_out = jnp.where(success, new_g, g)
            gs_out = jnp.where(success, gs, 0.0)

            descent_norm = gs_out * gradnorm(dd)

            # absdelta convergence (NIFTy optimize.py:407-414)
            min_cond = (ls_iter < 2) & (i > miniter)
            status = jnp.where(
                (energy_diff >= 0.0) & (energy_diff < 1e-3) & min_cond & (status != -1),
                jnp.int32(0),
                status,
            )
            # xtol convergence (NIFTy optimize.py:415-417)
            status = jnp.where(
                (descent_norm <= ncg_xtol) & (i > miniter) & (status != -1),
                jnp.int32(0),
                status,
            )
            # maxiter (NIFTy optimize.py:418)
            status = jnp.where((i == maxiter) & (status < -1), i, status)

            return (pos_out, energy_out, old_energy, g_out, status, i)

        result = jax.lax.while_loop(ncg_cond, ncg_body, init_state)
        return result[0], result[1]

    def curve_residual(m, r_linear, metric_key, sign, data_args):
        """Nonlinearly update a linear residual to a geoVI curved residual.

        Implements NIFTy's ``nonlinearly_update_residual`` algorithm exactly
        (evi.py:136-217) using ``_newton_cg_flat`` for the inner
        Newton-CG optimization.

        Parameters
        ----------
        m : flat array, expansion point
        r_linear : flat array, linear residual (covariance M^{-1})
        metric_key : PRNG key (same as used for draw_residuals)
        sign : +1.0 or -1.0 (for mirrored samples)
        data_args : dict, data-dependent arguments

        Returns
        -------
        flat array : curved residual (x_opt - m)
        """
        x0 = m + r_linear
        ms = sign * draw_metric_sample(m, metric_key, data_args)
        trafo_at_m = transformation_flat(m, data_args)

        def phi_vg(x):
            """Compute phi value and natural gradient for nonlinear residual curving."""
            trafo_x = transformation_flat(x, data_args)
            delta_trafo = trafo_x - trafo_at_m
            g_x = (x - m) + left_sqrt_metric_flat(m, delta_trafo, data_args)
            r = ms - g_x
            val = 0.5 * jnp.dot(r, r)
            ngrad = r + left_sqrt_metric_flat(
                x, right_sqrt_metric_flat(m, r, data_args), data_args
            )
            return val, -ngrad

        def phi_metric(x, v):
            """Compute phi metric-vector product for nonlinear curving."""
            tm = left_sqrt_metric_flat(m, right_sqrt_metric_flat(x, v, data_args), data_args) + v
            return (
                left_sqrt_metric_flat(x, right_sqrt_metric_flat(m, tm, data_args), data_args) + tm
            )

        # sampnorm (evi.py:178-181)
        def sampnorm(natgrad):
            """Compute sample norm used in Newton-CG for residual curving."""
            fpp = right_sqrt_metric_flat(m, natgrad, data_args)
            return jnp.sqrt(jnp.dot(natgrad, natgrad) + jnp.dot(fpp, fpp))

        x_opt, _ = _newton_cg_flat(
            phi_vg,
            phi_metric,
            x0,
            custom_gradnorm=sampnorm,
            maxiter=3,
            miniter=0,
            xtol=1e-3,
            energy_reduction_factor=0.1,
        )
        return x_opt - m

    def draw_nonlinear_residuals(m, subkeys, data_args):
        """Draw geoVI nonlinear residuals: linear draw + curving + mirror.

        Returns (2*n_samples, D) array: curved residuals with mirrored pairs.
        Matches NIFTy's ``nonlinear_resample`` sample mode.
        """
        # First draw linear residuals
        linear_residuals = draw_residuals(m, subkeys, data_args)

        # Curve each residual and its mirror
        def curve_pair(r, subkey):
            """Curve a linear residual and its mirror (both signs)."""
            r_pos = curve_residual(m, r, subkey, sign=1.0, data_args=data_args)
            r_neg = curve_residual(m, -r, subkey, sign=-1.0, data_args=data_args)
            return r_pos, r_neg

        pos_curved, neg_curved = jax.vmap(curve_pair)(linear_residuals, subkeys)
        return jnp.concatenate([pos_curved, neg_curved], axis=0)

    def update_nonlinear_residuals(m, prev_residuals, subkeys, data_args):
        """Re-curve existing residuals at updated expansion point.

        Takes 2*n_samples residuals (first half positive, second half
        negative mirrors) and re-applies geoVI curving at the new m.
        Matches NIFTy's ``nonlinear_update`` sample mode.
        """
        n_half = prev_residuals.shape[0] // 2
        r_pos = prev_residuals[:n_half]
        r_neg = prev_residuals[n_half:]

        def recurve_pair(r_p, r_n, subkey):
            """Re-curve a mirrored pair of residuals at new expansion point."""
            new_p = curve_residual(m, r_p, subkey, sign=1.0, data_args=data_args)
            new_n = curve_residual(m, r_n, subkey, sign=-1.0, data_args=data_args)
            return new_p, new_n

        new_pos, new_neg = jax.vmap(recurve_pair)(r_pos, r_neg, subkeys)
        return jnp.concatenate([new_pos, new_neg], axis=0)

    # --- EVI optimizer: fully JIT'd optimize_kl ---
    def kl_vg(m, residuals, data_args):
        """KL value and gradient averaged over samples."""

        def single_vg(r):
            """Compute Hamiltonian value and gradient for one residual."""
            return H_vg(m + r, data_args)

        vals, grads = jax.vmap(single_vg)(residuals)
        return jnp.mean(vals), jnp.mean(grads, axis=0)

    def kl_metric(m, residuals, v, data_args):
        """KL metric-vector product averaged over samples."""

        def single_met(r):
            """Apply metric for one residual sample."""
            return metric_vec(m + r, v, data_args)

        return jnp.mean(jax.vmap(single_met)(residuals), axis=0)

    def evi_step(m, subkey, n_samples, data_args):
        """One EVI iteration: draw samples + Newton-CG KL minimize.

        Returns (m_new, kl_value).
        """
        sample_keys = jax.random.split(subkey, n_samples)
        residuals = draw_residuals(m, sample_keys, data_args)
        residuals = jnp.concatenate([residuals, -residuals], axis=0)

        def _evi_kl_vg(m_cur):
            """Compute KL value and gradient at current point."""
            return kl_vg(m_cur, residuals, data_args)

        def _evi_kl_hessp(m_cur, v):
            """Compute KL Hessian-vector product at current point."""
            return kl_metric(m_cur, residuals, v, data_args)

        m_opt, kl_val = _newton_cg_flat(
            _evi_kl_vg,
            _evi_kl_hessp,
            m,
            maxiter=10,
            miniter=0,
            xtol=1e-3,
            energy_reduction_factor=0.1,
        )
        return m_opt, kl_val

    def run_evi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol):
        """Run EVI with automatic convergence detection via ``jax.lax.while_loop``."""

        def cond_fn(state):
            """Continue EVI if not converged and iterations remain."""
            _m, _prev_kl, i, converged = state
            return (~converged) & (i < n_iterations)

        def body_fn(state):
            """Execute one EVI iteration."""
            m, prev_kl, i, converged = state
            subkey = jax.random.fold_in(key, i)
            m_new, kl_val = evi_step(m, subkey, n_samples, data_args)
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            converged = (rel_change < kl_rtol) & (i >= 5)
            return (m_new, kl_val, i + 1, converged)

        first_key = jax.random.fold_in(key, 0)
        m0, kl0 = evi_step(init_pos, first_key, n_samples, data_args)
        init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))

        m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return m_final, n_iters

    # --- geoVI optimizer: per-mode functions (no lax.switch) ---
    #
    # Each sample mode gets its own evi_step function so that JAX
    # compiles ONLY the code path actually used.  This avoids the
    # 56s compilation cost of tracing all three branches via
    # ``jax.lax.switch``.
    #
    # ``sample_mode`` is a **static** string argument: JAX caches
    # a separate compiled version for each mode.
    SAMPLE_LINEAR = jnp.int32(0)
    SAMPLE_NONLINEAR_RESAMPLE = jnp.int32(1)
    SAMPLE_NONLINEAR_UPDATE = jnp.int32(2)

    def _kl_minimize(m, residuals, constants_mask, data_args):
        """Newton-CG KL minimization with constants mask."""

        def _masked_kl_vg(m_cur, res):
            """Compute KL value and gradient with constants masked out."""
            val, grad = kl_vg(m_cur, res, data_args)
            grad = jnp.where(constants_mask, 0.0, grad)
            return val, grad

        def _masked_kl_metric(m_cur, res, v):
            """Apply KL metric with constants masked out."""
            v_masked = jnp.where(constants_mask, 0.0, v)
            mv = kl_metric(m_cur, res, v_masked, data_args)
            return jnp.where(constants_mask, 0.0, mv)

        def _fun_and_grad(m_cur):
            """Compute masked KL value and gradient."""
            return _masked_kl_vg(m_cur, residuals)

        def _hessp(m_cur, v):
            """Apply masked KL Hessian-vector product."""
            return _masked_kl_metric(m_cur, residuals, v)

        return _newton_cg_flat(
            _fun_and_grad,
            _hessp,
            m,
            maxiter=10,
            miniter=0,
            xtol=1e-3,  # match NIFTy default (vi_config.py)
            energy_reduction_factor=0.1,
        )

    _RESAMPLE_EVERY = 5  # refresh stale samples every N iterations

    def evi_step_full(
        m,
        subkey,
        n_samples,
        sample_mode,
        prev_residuals,
        prev_keys,
        constants_mask,
        pe_mask,
        data_args,
        iteration=0,
    ):
        """One geoVI iteration — ``sample_mode`` must be a static string.

        When used inside ``run_evi_geovi`` (which marks ``sample_mode``
        as static), JAX compiles a separate version per mode.  The
        unused branches are never traced, so ``"linear"`` compiles in
        ~0.03s while ``"nonlinear_resample"`` compiles in ~56s.

        Parameters
        ----------
        sample_mode : str  (STATIC — triggers recompilation per value)
            ``"linear_resample"`` — fresh MGVI samples (standard MGVI)
            ``"linear_sample"`` — reuse keys from prev iter (deterministic MGVI)
            ``"nonlinear_resample"`` — fresh geoVI samples (standard geoVI)
            ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
            ``"nonlinear_update"`` — re-curve existing residuals at new m
        data_args : dict
            Data-dependent arguments (data, noise, sqrt_noise_inv, etc.).

        Returns
        -------
        m_new, kl_value, new_residuals, used_keys
        """
        # Key handling: _resample = fresh keys, _sample = reuse prev keys
        if sample_mode.endswith("_resample") or sample_mode == "vi":
            sample_keys = jax.random.split(subkey, n_samples)
        elif sample_mode == "nonlinear_update":
            sample_keys = prev_keys
        else:  # _sample modes: reuse
            sample_keys = prev_keys

        # Python if — only the used branch is traced by JAX
        if sample_mode == "vi":
            # Optimal schedule: resample at iter 0 and every
            # _RESAMPLE_EVERY, nonlinear_update in between.
            # Uses jax.lax.cond (traces both branches, executes one).
            do_resample = (iteration == 0) | (iteration % _RESAMPLE_EVERY == 0)

            def _do_resample(_):
                """Draw fresh nonlinear residuals."""
                return draw_nonlinear_residuals(m, sample_keys, data_args)

            def _do_update(_):
                """Re-curve existing residuals at new expansion point."""
                return update_nonlinear_residuals(m, prev_residuals, prev_keys, data_args)

            residuals = jax.lax.cond(do_resample, _do_resample, _do_update, None)
        elif sample_mode in ("nonlinear_resample", "nonlinear_sample"):
            residuals = draw_nonlinear_residuals(m, sample_keys, data_args)
        elif sample_mode == "nonlinear_update":
            residuals = update_nonlinear_residuals(m, prev_residuals, sample_keys, data_args)
        else:  # linear_resample, linear_sample
            res = draw_residuals(m, sample_keys, data_args)
            residuals = jnp.concatenate([res, -res], axis=0)

        # Apply point estimates mask
        residuals = residuals * pe_mask[None, :]

        # KL minimization
        m_opt, kl_val = _kl_minimize(m, residuals, constants_mask, data_args)
        return m_opt, kl_val, residuals, sample_keys

    def run_evi_geovi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol, sample_mode):
        """Run geoVI with automatic convergence detection.

        ``n_iterations`` is a **dynamic** traced value — changing it
        does NOT trigger recompilation.  Keys are generated on-the-fly
        via ``jax.random.fold_in`` instead of pre-splitting.

        ``sample_mode`` is a **static** string — JAX compiles a
        separate XLA program per mode.  All 5 NIFTy modes supported:

        - ``"linear_resample"`` — fresh MGVI samples each iteration
        - ``"linear_sample"`` — reuse PRNG keys (deterministic MGVI)
        - ``"nonlinear_resample"`` — fresh geoVI samples
        - ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
        - ``"nonlinear_update"`` — re-curve existing residuals at new m

        """
        # Generate per-iteration keys on-the-fly via fold_in (no
        # pre-split needed, so n_iterations can be dynamic).
        dummy_residuals = jnp.zeros((2 * n_samples, d_total))
        dummy_keys = jax.random.split(jax.random.fold_in(key, 0), n_samples)
        no_constants = jnp.zeros(d_total, dtype=bool)
        all_sampled = jnp.ones(d_total)

        # State: (m, prev_kl, residuals, prev_keys, iter, converged)
        def cond_fn(state):
            """Continue geoVI if not converged and iterations remain."""
            _m, _prev_kl, _res, _pk, i, converged = state
            return (~converged) & (i < n_iterations)

        def body_fn(state):
            """Execute one geoVI iteration."""
            m, prev_kl, prev_res, prev_k, i, converged = state
            subkey = jax.random.fold_in(key, i)
            m_new, kl_val, new_res, new_k = evi_step_full(
                m,
                subkey,
                n_samples,
                sample_mode,
                prev_res,
                prev_k,
                no_constants,
                all_sampled,
                data_args,
                iteration=i,
            )
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            converged = (rel_change < kl_rtol) & (i >= 5)
            return (m_new, kl_val, new_res, new_k, i + 1, converged)

        # First iteration (always resample to establish initial keys)
        first_key = jax.random.fold_in(key, 0)
        m0, kl0, res0, keys0 = evi_step_full(
            init_pos,
            first_key,
            n_samples,
            sample_mode,
            dummy_residuals,
            dummy_keys,
            no_constants,
            all_sampled,
            data_args,
        )
        init_state = (
            m0,
            kl0,
            res0,
            keys0,
            jnp.int32(1),
            jnp.bool_(False),
        )

        result = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return result[0], result[4]  # m_final, n_iters

    # --- Parameter range mapping for mask construction ---
    param_ranges = {}
    for i_k, k in enumerate(param_keys):
        start, end, _shape = slices[i_k]
        param_ranges[k] = (start, end)

    def make_mask(param_names):
        """Create boolean mask: True for named params, False otherwise."""
        mask = jnp.zeros(d_total, dtype=bool)
        for name in param_names:
            if name in param_ranges:
                start, end = param_ranges[name]
                mask = mask.at[start:end].set(True)
        return mask

    def make_pe_mask(param_names):
        """Create point-estimate mask: 0.0 for PE params, 1.0 for sampled."""
        mask = jnp.ones(d_total)
        for name in param_names:
            if name in param_ranges:
                start, end = param_ranges[name]
                mask = mask.at[start:end].set(0.0)
        return mask

    # --- NIFTy model (physics wrapper, no data) ---
    # ``_nifty_model`` wraps the shared signal_response_jit so NIFTy's trace
    # cache hits on the second galaxy.  The per-Fitter nifty_likelihood
    # (which captures galaxy data) is built in run_nifty_fast_vi via
    # _get_or_build_nifty_likelihood — NOT here, so the shared engine
    # never holds a stale reference to any galaxy's flux/noise arrays.
    _nifty_model = None
    if _has_nifty:
        try:
            import nifty8.re as jft

            _, _sr_jit = get_or_build_signal_response(fitter)
            _nifty_domain = {}
            for name in fitter._free_names:
                _nifty_domain[name] = jft.ShapeWithDtype(())
            if fitter.spec.stochastic:
                _nifty_domain["psd_xi"] = jft.ShapeWithDtype((fitter.spec.n_grid,))
            _nifty_model = jft.Model(_sr_jit, domain=_nifty_domain)
        except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError):
            _has_nifty = False
            _nifty_model = None

    # Wrap core functions in JIT but do NOT pre-compile (no dummy calls).
    # Compilation happens lazily on first real call — avoids the 2+ GB
    # protobuf size limit that eager compilation can hit when the forward
    # model is large.  signal_response is already JIT'd above so it
    # won't be re-traced into these scopes.
    from tengri.utils.compile_log import instrument_first_call

    sig = fitter.compile_signature()
    draw_samples_jit = instrument_first_call(
        jax.jit(draw_residuals), "draw_samples", sig, method="vi"
    )
    run_evi_jit = instrument_first_call(
        jax.jit(run_evi, static_argnames=("n_samples",)),
        "run_evi",
        sig,
        method="vi",
    )
    run_evi_geovi_jit = instrument_first_call(
        jax.jit(
            run_evi_geovi,
            static_argnames=("n_samples", "sample_mode"),
        ),
        "run_evi_geovi",
        sig,
        method="geovi",
    )
    draw_nonlinear_jit = instrument_first_call(
        jax.jit(draw_nonlinear_residuals),
        "draw_nonlinear_samples",
        sig,
        method="vi",
    )

    return {
        "run_evi": run_evi_jit,
        "native_vi_linear_run": run_evi_jit,  # canonical name (data_args variant)
        "run_evi_geovi": run_evi_geovi_jit,
        "nifty_model": _nifty_model,
        "draw_samples": draw_samples_jit,
        "native_vi_linear_draw": draw_samples_jit,  # canonical name
        "draw_nonlinear_samples": draw_nonlinear_jit,
        "draw_batch": draw_batch,
        "flatten": flatten,
        "unflatten": unflatten,
        "param_keys": param_keys,
        "param_ranges": param_ranges,
        "make_mask": make_mask,
        "make_pe_mask": make_pe_mask,
        "d_total": d_total,
        "SAMPLE_LINEAR": SAMPLE_LINEAR,
        "SAMPLE_NONLINEAR_RESAMPLE": SAMPLE_NONLINEAR_RESAMPLE,
        "SAMPLE_NONLINEAR_UPDATE": SAMPLE_NONLINEAR_UPDATE,
        "evi_step_full": evi_step_full,
        # geoVI-NUTS primitives (coordinate transform + metric)
        "transformation_flat": transformation_flat,
        "left_sqrt_metric_flat": left_sqrt_metric_flat,
        "right_sqrt_metric_flat": right_sqrt_metric_flat,
        "metric_vec": metric_vec,
        "cg_solve": cg_solve,
        "hamiltonian": hamiltonian,
    }
