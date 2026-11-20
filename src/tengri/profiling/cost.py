# SPDX-License-Identifier: BSD-3-Clause
"""Compiled-HLO FLOP counts for tengri.

The one shared FLOP-count path for benchmarks and tests. Work is compared by
the FLOPs XLA reports for the compiled graph, which is machine-independent,
unlike a wall clock.

Usage
-----
>>> from tengri.profiling import compiled_flops
>>> flops = compiled_flops(jax.grad(loss), params)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax


def compiled_flops(fn: Callable[..., Any], *args: Any) -> int | None:
    """Return the compiled-HLO FLOP count of ``fn`` at ``args``.

    Parameters
    ----------
    fn : callable
        Function to compile. It is wrapped in ``jax.jit`` here.
    *args : array_like
        Example arguments fixing the shapes and dtypes to compile for.

    Returns
    -------
    int or None
        The ``"flops"`` entry of ``cost_analysis()``, or ``None`` when the
        backend returns an empty analysis or one with no ``"flops"`` key.

    Raises
    ------
    Exception
        Any error from lowering or compiling ``fn`` propagates; it is not
        converted to ``None``.

    Notes
    -----
    Compiles ``fn`` once per call and does not execute it. Accepts both
    return shapes ``cost_analysis()`` has used across JAX releases: a dict,
    or a list of dicts (one per computation), of which the first is taken.
    Does not itself run under ``jax.jit``, ``jax.vmap`` or ``jax.grad``, so
    it is not traced; pass the transformed callable as ``fn`` instead.
    """
    analysis = jax.jit(fn).lower(*args).compile().cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else None
    if not analysis or "flops" not in analysis:
        return None
    return int(analysis["flops"])
