# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a per-draw model map over a posterior in memory-bounded chunks.

The posterior plots need one model evaluation per draw. A Python loop over the
draws pays per-call dispatch for every draw; :func:`map_draws` runs the same
evaluation through :func:`tengri.utils.batching.vmap_chunked` instead, so each
chunk is one compiled kernel and peak memory scales with the chunk.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np

from tengri.utils.batching import vmap_chunked

__all__ = ["DRAW_CHUNK", "map_draws"]

#: Draws per compiled chunk for posterior plots.
DRAW_CHUNK = 16


def map_draws(fn: Callable, samples: Mapping) -> np.ndarray:
    """Apply ``fn`` to every posterior draw and stack the results.

    Equivalent to ``np.array([fn({k: v[i] for k, v in samples.items()}) for i in
    range(n)])``, evaluated through :func:`~tengri.utils.batching.vmap_chunked`.

    Parameters
    ----------
    fn : callable
        ``dict -> array``, one draw's parameters to one array. Pass a stable
        callable (a bound method of the model) so its compiled kernel is cached.
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
    batch = {k: jnp.asarray(v) for k, v in samples.items()}
    return np.asarray(vmap_chunked(fn, chunk_size=DRAW_CHUNK)(batch))
