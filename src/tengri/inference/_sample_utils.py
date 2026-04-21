"""Shared post-processing utilities for all inference backends.

Every sampler needs the same three operations after collecting samples:
convert flat unbounded positions → physical dicts, summarise with a
point estimate, and optionally warm-start from a MAP.  Centralising
them here removes ~50 duplicated blocks across the backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

_MAP_INIT_STEPS = 200


def _vmap_samples_to_physical(
    samples: Array,
    unravel_fn: Callable[[Array], dict],
    to_physical_fn: Callable[[dict], dict],
) -> dict[str, Array]:
    """Convert a batch of flat unbounded samples to physical parameter dicts."""

    def _convert_one(flat_pos: Array) -> dict[str, Array]:
        """Convert single flat unbounded sample to physical parameter dict."""
        return to_physical_fn(unravel_fn(flat_pos))

    return jax.vmap(_convert_one)(samples)


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
