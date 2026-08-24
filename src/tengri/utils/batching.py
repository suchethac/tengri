# SPDX-License-Identifier: BSD-3-Clause
r"""Memory-bounded batched evaluation: :func:`vmap_chunked`.

A bare :func:`jax.vmap` over a full posterior materializes every intermediate
for every draw at once, which is how a 20 000-sample SED fit turns into an OOM
with no obvious culprit. A bare Python loop avoids the memory blow-up but gives
up the batched kernel and runs ~7x slower.

:func:`vmap_chunked` is the middle path every production notebook was
reinventing: ``jax.jit(jax.vmap(fn))`` applied over slices of the batch, with
the results concatenated along the draw axis. Peak memory scales with
``chunk_size`` rather than with the number of draws.

Notes
-----
**Correctness contract**: the result must not depend on ``chunk_size`` beyond a
few ULP. Chunking is a memory strategy, not a numerical one; a chunk-boundary
bug is dangerous precisely because it produces plausible numbers rather than an
error.

**Chunking is not bit-neutral, though.** XLA compiles a *different kernel for
each batch shape*, so a chunk of 8 and a batch of 37 vectorize and reassociate
their reductions differently and the last bit can move (measured: ~1 ULP on
``sfr_100myr``; exact on ``stellar_mass``). If you are chasing bit-for-bit
reproducibility (a reproduction notebook, a parity audit) pin ``chunk_size``
along with everything else, or evaluate unchunked.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

__all__ = ["vmap_chunked"]

# The failures that genuinely mean "this function cannot be traced": it inspects
# concrete values, or leaks a tracer. Everything else (a typo'd key, a shape
# mismatch, an OOM) is a real bug and must propagate: catching it here would
# silently reclassify it as a fact of life and route around it forever (#1128).
_NOT_TRACEABLE = (
    jax.errors.ConcretizationTypeError,  # base of the Tracer*ConversionError family
    jax.errors.UnexpectedTracerError,
    jax.errors.NonConcreteBooleanIndexError,
)


def _leading_axis_size(batch) -> int:
    """Length of the batch (draw) axis, read from the first leaf."""
    leaves = jax.tree_util.tree_leaves(batch)
    if not leaves:
        raise ValueError("empty batch: nothing to map over")
    return leaves[0].shape[0]


def _slice_batch(batch, start: int, stop: int):
    """One contiguous slice of every leaf along the draw axis."""
    return jax.tree_util.tree_map(lambda x: x[start:stop], batch)


def vmap_chunked(fn, chunk_size: int = 16):
    r"""Map ``fn`` over a batch in memory-bounded chunks.

    Equivalent to ``jax.vmap(fn)`` in its result, but evaluates the batch
    ``chunk_size`` draws at a time so peak memory scales with the chunk rather
    than with the batch. Each chunk runs as one compiled ``jit(vmap(fn))``
    kernel.

    If ``fn`` cannot be jitted (some nebular backends are not traceable) the
    call degrades to an **eager per-draw loop** rather than raising, and warns
    once so the ~7x slowdown is never silent. The jittability probe runs **once**
    per returned callable, not once per draw.

    Only genuine *tracing* failures count as "not jittable". A typo'd key, a shape
    mismatch, or any other real bug propagates rather than being quietly
    reclassified and routed around (#1128).

    Parameters
    ----------
    fn : callable
        A function of a single parameter pytree (e.g. ``params -> dict``). It is
        mapped over the leading axis of every leaf of the batch.
    chunk_size : int, default 16
        Draws per compiled call. Larger is faster and hungrier. Values larger
        than the batch are harmless (one chunk).

    Returns
    -------
    callable
        ``batch -> results``, where ``batch`` is a pytree whose leaves share a
        leading draw axis of length ``n``, and each result leaf has leading axis
        ``n``.

    Warns
    -----
    UserWarning
        Once per callable, when ``fn`` turns out not to be traceable and the
        evaluation falls back to the eager per-draw loop.

    Raises
    ------
    ValueError
        If ``chunk_size`` is not positive, or the batch has no leaves.
    Exception
        Whatever ``fn`` raises, if the failure is not a tracing failure; a real
        bug is never swallowed by the jittability probe.

    Notes
    -----
    **JIT-compatible**: the returned callable is *not* itself meant to be jitted;
    it drives compilation internally and does Python-level slicing. Use it
    around a jittable ``fn``, not inside another ``jit``.

    The ragged final chunk (when ``n`` is not a multiple of ``chunk_size``)
    triggers one extra, cheap compile for its distinct shape.

    **Not bit-neutral.** XLA compiles a different kernel per batch shape, so
    changing ``chunk_size`` can move the last bit (~1 ULP). Pin it if you need
    bit-for-bit reproducibility.

    Examples
    --------
    >>> from tengri import vmap_chunked
    >>> masses = vmap_chunked(  # doctest: +SKIP
    ...     lambda p: model.predict_properties(p, names=("stellar_mass",)),
    ...     chunk_size=32,
    ... )(posterior.samples)
    >>> masses["stellar_mass"].shape  # doctest: +SKIP
    (4000,)
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    jitted = jax.jit(jax.vmap(fn))
    # Tri-state: None = not yet probed; True/False = the settled answer. The
    # probe is per-callable, so a non-jittable fn costs ONE failed trace, not
    # one per draw.
    can_jit: list[bool | None] = [None]

    def _eager(batch, n):
        """Per-draw Python loop: the fallback for non-traceable ``fn``.

        ``jax.vmap`` *removes* the mapped axis before calling ``fn``, so ``fn``
        sees a scalar per draw. The loop must index (``x[i]``), not slice
        (``x[i : i + 1]``), or the fallback hands ``fn`` a shape-``(1,)`` array
        where the fast path hands it a scalar. That difference is invisible for
        a jittable ``fn`` (everything broadcasts) and fatal for exactly the
        concrete-inspecting functions this fallback exists to serve.
        """
        parts = [fn(jax.tree_util.tree_map(lambda x, i=i: x[i], batch)) for i in range(n)]
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *parts)

    def mapped(batch):
        n = _leading_axis_size(batch)

        if can_jit[0] is None:
            try:
                jitted(_slice_batch(batch, 0, min(chunk_size, n)))
                can_jit[0] = True
            except _NOT_TRACEABLE as exc:
                can_jit[0] = False
                warnings.warn(
                    f"vmap_chunked: {getattr(fn, '__name__', 'the mapped function')} "
                    f"cannot be traced ({type(exc).__name__}), so it will be evaluated "
                    f"one draw at a time in an eager loop; roughly 7x slower than the "
                    f"batched path. This is expected for backends that inspect concrete "
                    f"values; if you did not intend it, make the function jittable.",
                    UserWarning,
                    stacklevel=2,
                )

        if not can_jit[0]:
            return _eager(batch, n)

        parts = [
            jitted(_slice_batch(batch, s, min(s + chunk_size, n))) for s in range(0, n, chunk_size)
        ]
        if len(parts) == 1:
            return parts[0]
        return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)

    return mapped
