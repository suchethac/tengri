# SPDX-License-Identifier: BSD-3-Clause
"""Shared post-processing utilities for all inference backends.

Every sampler needs the same three operations after collecting samples:
convert flat unbounded positions → physical dicts, summarise with a
point estimate, and optionally warm-start from a MAP.  Centralising
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

_MAP_INIT_STEPS = 200

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

    If init_from is given, converts it to unbounded coordinates.
    Otherwise runs a short MAP optimisation to find a good starting point.
    """
    if init_from is not None:
        return fitter._unbounded_from_posterior(init_from), key
    if verbose:
        print(f"  MAP initialization ({n_map_steps} steps)...")
    key, map_key = jax.random.split(key)
    map_result = fitter._run_map(key=map_key, n_steps=n_map_steps, verbose=False)
    init_params = fitter._unbounded_from_posterior(map_result)
    if verbose:
        print(f"  MAP init done (loss={map_result.diagnostics['final_loss']:.2f})")
    return init_params, key
