# SPDX-License-Identifier: BSD-3-Clause
"""Shared post-processing utilities for all inference backends.

Every sampler needs the same three operations after collecting samples:
convert flat unbounded positions → physical dicts, summarize with a
point estimate, and optionally warm-start from a MAP.  Centralizing
them here removes ~50 duplicated blocks across the backends.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

_MAP_INIT_STEPS = 1000
#: Number of vmap'd ADAM restarts for the MCMC MAP-init seed. The standardized
#: prior is genuinely uniform, so a single random init can land in a poor basin;
#: a handful of parallel restarts (keep-best) seeds all chains at a good point so
#: NUTS/HMC converge in a short warm-up. Kept modest so the seed stays ~1 s.
_MAP_INIT_RESTARTS = 8

# Bound the cache so a long session iterating over many fitters does not
# accumulate compiled vmaps unboundedly. LRU-evict on insert.
_VMAP_TO_PHYSICAL_CACHE: OrderedDict = OrderedDict()
_VMAP_TO_PHYSICAL_CACHE_LOCK = Lock()
_VMAP_TO_PHYSICAL_CACHE_MAX = 8


def _vmap_samples_to_physical(
    samples: Array,
    unravel_fn: Callable[[Array], dict],
    to_physical_fn: Callable[[dict], dict],
) -> dict[str, Array]:
    """Convert a batch of flat unbounded samples to physical parameter dicts.

    The compiled vmap is cached on the fitter (recovered from the bound
    method's ``__self__``) keyed on ``(id(fitter), n_dim)``. Without
    this cache every ``Fitter.run`` re-traces and recompiles the
    conversion (~700 ms photometry, more on spec). ``unravel_fn``
    captured by the cache entry is the one from the first call —
    structurally identical to subsequent ones for the same fitter, so
    reusing it is safe.
    """
    fitter = getattr(to_physical_fn, "__self__", None)
    if fitter is None:
        # No bound-method fitter → can't cache safely; fall back to per-call vmap.
        return jax.vmap(lambda p: to_physical_fn(unravel_fn(p)))(samples)

    n_dim = int(samples.shape[-1])
    key = (id(fitter), n_dim)
    with _VMAP_TO_PHYSICAL_CACHE_LOCK:
        cached = _VMAP_TO_PHYSICAL_CACHE.get(key)
        if cached is None:
            # Capture the unravel_fn closure from THIS call; structurally
            # equivalent to any future unravel_fn for the same fitter.
            unravel_captured = unravel_fn
            to_phys_captured = to_physical_fn

            @jax.jit
            def _convert_one(flat_pos: Array) -> dict[str, Array]:
                return to_phys_captured(unravel_captured(flat_pos))

            cached = jax.vmap(_convert_one)
            _VMAP_TO_PHYSICAL_CACHE[key] = cached
            while len(_VMAP_TO_PHYSICAL_CACHE) > _VMAP_TO_PHYSICAL_CACHE_MAX:
                _VMAP_TO_PHYSICAL_CACHE.popitem(last=False)
        else:
            _VMAP_TO_PHYSICAL_CACHE.move_to_end(key)
    return cached(samples)


def _clear_vmap_to_physical_cache() -> None:
    """Drop the cached vmap conversions (called by ``tengri.gc()``)."""
    with _VMAP_TO_PHYSICAL_CACHE_LOCK:
        _VMAP_TO_PHYSICAL_CACHE.clear()


def _mean_params(samples_phys: dict[str, Array]) -> dict[str, Array]:
    """Return per-parameter posterior means."""
    return {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}


def _maybe_map_init(
    fitter: Any,
    key: Array,
    init_from: Any,
    verbose: bool,
    n_map_steps: int = _MAP_INIT_STEPS,
) -> tuple[dict, Array]:
    """Return (init_params_unbounded, updated_key).

    Resolution order:

    1. Explicit ``init_from`` from caller — converted to unbounded.
    2. Cached MAP point on the model (populated by a previous fit or by
       :meth:`Fitter.load_cache`) — used directly, no fresh MAP run.
    3. Run a short MAP optimization to find a good starting point.
       Caches the result so subsequent calls / sessions can skip step 3.
    """
    if init_from is not None:
        return fitter._unbounded_from_posterior(init_from), key

    # Cached MAP point (from a prior fit or load_cache)?

    from tengri.inference._model_cache import _default_owner as _model_cache_owner

    mc = _model_cache_owner.get_or_compile_model(fitter.model)
    cached_map = mc.get("map_params_physical")
    if cached_map is not None:
        # Build a minimal posterior-like shim to reuse _unbounded_from_posterior.
        class _Shim:
            def __init__(self, params):
                self.params = params

        if verbose:
            print("  MAP initialization: reusing cached MAP point")
        return fitter._unbounded_from_posterior(_Shim(cached_map)), key

    # Pre-warm: one eager forward call resolves a tracing pathology in
    # ``Uniform.unstandardize`` that crashes the MAP-init JIT trace on
    # models whose forward pass has never been exercised in Python (e.g.
    # ``Fitter(model, flux_from_disk, noise).run('mcmc_hmc')`` with no
    # prior ``model.predict_photometry(...)`` or ``model.mock(...)``).
    # See issue #262 for the narrow reproducer and traceback.
    _prewarm_forward(fitter)
    if verbose:
        print(f"  MAP initialization ({n_map_steps} steps)...")
    key, map_key = jax.random.split(key)
    map_result = fitter._run_map(
        key=map_key, n_steps=n_map_steps, n_restarts=_MAP_INIT_RESTARTS, verbose=False
    )
    # Cache the physical MAP params so future runs/sessions skip MAP.
    mc["map_params_physical"] = {k: jnp.asarray(v) for k, v in map_result.params.items()}
    init_params = fitter._unbounded_from_posterior(map_result)
    if verbose:
        print(f"  MAP init done (loss={map_result.diagnostics['final_loss']:.2f})")
    return init_params, key


def _prewarm_forward(fitter: Any) -> None:
    """Run one eager forward pass to materialize lazy unbounded→physical state.

    Hit by issue #262: when ``Fitter`` is constructed against externally
    sourced data (e.g. ``np.load(npz)``) and the model has never had a
    Python-side forward pass on it, the first JIT trace through
    ``Uniform.unstandardize`` keeps a traced array escaping into
    ``float()``. Calling ``predict_photometry`` once eagerly before
    JIT tracing materializes the relevant arrays and resolves the
    leak. Best-effort: if the call fails for any other reason, swallow
    silently and let the real path raise.
    """
    try:
        sample_params = {}
        for name in fitter._free_names:
            dist = fitter.spec.get_distribution(name)
            if hasattr(dist, "low") or hasattr(dist, "lo"):
                lo = getattr(dist, "low", getattr(dist, "lo", 0.0))
                hi = getattr(dist, "high", getattr(dist, "hi", 1.0))
                sample_params[name] = float(0.5 * (lo + hi))
            elif hasattr(dist, "mean"):
                sample_params[name] = float(dist.mean)
            else:
                sample_params[name] = 0.0
        for name, value in fitter._fixed_values.items():
            sample_params[name] = float(value)
        fitter.model.predict_photometry(sample_params)
    except Exception:
        # Pre-warm is a soft guarantee; the real path will surface
        # any genuine failure with a richer traceback.
        pass
