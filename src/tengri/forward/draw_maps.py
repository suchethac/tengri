# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a per-draw model map over a posterior in memory-bounded chunks.

The posterior plots need one model evaluation per draw. A Python loop over the
draws pays per-call dispatch for every draw; these helpers run the same
evaluation through :func:`tengri.utils.batching.vmap_chunked` instead, so each
chunk is one compiled kernel and peak memory scales with the chunk.

Two entry points. :func:`map_draws` is for a :class:`~tengri.forward.sed_model.SEDModel`
and caches its compiled kernel on the model, keyed by method name and compile
signature: a repeated plot reuses the program, and a structural change to the
model recompiles. :func:`evaluate_draws` takes any callable and compiles per call.
Nothing global references a model, so a model can be garbage-collected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np

from tengri.utils.batching import vmap_chunked

__all__ = ["DRAW_CHUNK", "evaluate_draws", "map_draws"]

#: Draws per compiled chunk for posterior plots.
DRAW_CHUNK = 16


def _batch(samples: Mapping) -> dict:
    """Samples as a pytree of jax arrays, each with the draw axis first."""
    return {k: jnp.asarray(v) for k, v in samples.items()}


def evaluate_draws(fn: Callable, samples: Mapping) -> np.ndarray:
    """Apply ``fn`` to every posterior draw and stack the results (uncached).

    Equivalent to ``np.array([fn({k: v[i] for k, v in samples.items()}) for i in
    range(n)])``, evaluated through :func:`~tengri.utils.batching.vmap_chunked`.
    Compiles ``fn`` on each call; use :func:`map_draws` for a model.

    Parameters
    ----------
    fn : callable
        ``dict -> array``, one draw's parameters to one array.
    samples : Mapping
        Parameter name to array, shape ``(n,)`` or ``(n, ...)``. Every entry has
        the draw axis first.

    Returns
    -------
    ndarray, shape (n, ...)
        Per-draw outputs of ``fn``.

    Notes
    -----
    **JIT-compatible** when ``fn`` is: a non-traceable ``fn`` falls back to the
    per-draw loop inside ``vmap_chunked``, with its warning.
    """
    return np.asarray(vmap_chunked(fn, chunk_size=DRAW_CHUNK)(_batch(samples)))


def map_draws(model, name: str, samples: Mapping) -> np.ndarray:
    """Apply the model method ``name`` to every posterior draw (cached on the model).

    The compiled kernel is stored on ``model`` under ``name`` together with
    ``model.compile_signature()``. A later call with the same signature reuses it;
    a changed signature replaces it. One entry per method name, so the cache is
    bounded, and it lives on the model, so it is collected with the model.

    Parameters
    ----------
    model : SEDModel
        Model whose bound method ``name`` maps one draw's parameters to an array.
    name : str
        Method name, e.g. ``"_sfr_mean_draw"``.
    samples : Mapping
        Parameter name to array with the draw axis first.

    Returns
    -------
    ndarray, shape (n, ...)
        Per-draw outputs of ``model.<name>``.

    Notes
    -----
    **JIT-compatible**: the method is compiled per chunk of :data:`DRAW_CHUNK`
    draws. The cache key must cover everything the trace reads from the model,
    which is what ``compile_signature()`` promises.
    """
    return np.asarray(_owner_kernel(model, name)(_batch(samples)))


def _owner_kernel(model, name: str) -> Callable:
    """The mapped callable for ``model.<name>``, cached on ``model`` by signature."""
    signature = model.compile_signature()
    cache = vars(model).setdefault("_draw_kernels", {})
    hit = cache.get(name)
    if hit is not None and hit[0] == signature:
        return hit[1]
    mapped = vmap_chunked(getattr(model, name), chunk_size=DRAW_CHUNK)
    cache[name] = (signature, mapped)
    return mapped
