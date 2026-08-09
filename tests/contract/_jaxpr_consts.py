# SPDX-License-Identifier: BSD-3-Clause
"""Recursive jaxpr-constant accounting for the template-threading contract.

A threading guard has to count the arrays XLA will emit as ``Constant`` ops.
``ClosedJaxpr.consts`` alone is **not** that count: it is top-level only, so
anything an inner :func:`jax.jit` closed over lives in a sub-jaxpr and is
invisible there. That blind spot is why the check prescribed in #1201 reported
``0.00 MB`` against an HLO carrying 267 MB of SSP grid.

The walk below recurses through ``eqn.params`` (where ``jit``/``scan``/``cond``
stash their sub-jaxprs) and also collects array-valued ``Literal`` invars.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal

__all__ = ["iter_baked_arrays", "baked_bytes", "baked_mb"]


def _iter_sub_jaxprs(value: Any) -> Iterator[Jaxpr]:
    """Yield every ``Jaxpr`` nested anywhere inside an equation parameter."""
    if isinstance(value, ClosedJaxpr):
        yield value.jaxpr
    elif isinstance(value, Jaxpr):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_sub_jaxprs(item)


def iter_baked_arrays(closed_jaxpr: ClosedJaxpr, *, min_ndim: int = 1) -> Iterator[np.ndarray]:
    """Yield every constant array reachable from ``closed_jaxpr``.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        Traced graph, e.g. from ``jax.make_jaxpr(fn)(*args)``.
    min_ndim : int, optional
        Skip constants below this rank. Default ``1`` drops the scalar
        literals every graph carries, which are noise at MB scale.

    Yields
    ------
    ndarray
        Each baked constant, once per occurrence (an array inlined twice
        is genuinely paid twice in the HLO, so occurrences are not
        de-duplicated).
    """
    seen_jaxprs: set[int] = set()

    def _walk(jaxpr: Jaxpr, consts: tuple) -> Iterator[np.ndarray]:
        if id(jaxpr) in seen_jaxprs:
            return
        seen_jaxprs.add(id(jaxpr))
        for const in consts:
            arr = np.asarray(const)
            if arr.ndim >= min_ndim:
                yield arr
        for eqn in jaxpr.eqns:
            for invar in eqn.invars:
                if isinstance(invar, Literal):
                    arr = np.asarray(invar.val)
                    if arr.ndim >= min_ndim:
                        yield arr
            for param in eqn.params.values():
                for sub in _iter_sub_jaxprs(param):
                    sub_consts = getattr(param, "consts", ())
                    yield from _walk(sub, tuple(sub_consts))

    yield from _walk(closed_jaxpr.jaxpr, tuple(closed_jaxpr.consts))


def baked_bytes(closed_jaxpr: ClosedJaxpr, *, min_ndim: int = 1) -> int:
    """Total bytes of constants baked into ``closed_jaxpr``."""
    return sum(int(arr.nbytes) for arr in iter_baked_arrays(closed_jaxpr, min_ndim=min_ndim))


def baked_mb(closed_jaxpr: ClosedJaxpr, *, min_ndim: int = 1) -> float:
    """Total megabytes of constants baked into ``closed_jaxpr``."""
    return baked_bytes(closed_jaxpr, min_ndim=min_ndim) / (1024.0 * 1024.0)
