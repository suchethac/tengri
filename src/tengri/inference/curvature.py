# SPDX-License-Identifier: BSD-3-Clause
r"""Value, gradient and Hessian of a scalar log-density: the one place that differentiates twice.

Every consumer of curvature (the Laplace expansion, the damped Newton polish, the
preconditioning metric, the MAP stationarity certificate, the hierarchical mode search)
calls :func:`curvature`, so there is one second-derivative implementation and one
statement of what a non-finite result means. The helper works in float32 and float64
alike: the Hessian of the standardized posterior at its MAP is finite in both dtypes and
agrees to ``max|dH| / max|H| <= 1e-3`` (``tests/regression/precision/
test_dtype_curvature_sweep.py``, #3505).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["Curvature", "curvature"]

_FIELDS = frozenset({"value", "grad", "hess"})


class Curvature(NamedTuple):
    """Derivatives of a scalar function at one point.

    Attributes
    ----------
    value : jax.Array or None
        ``fn(x, *args)``, scalar; None when not requested.
    grad : jax.Array or None
        Gradient, shape ``(n_dim,)``; None when not requested.
    hess : jax.Array or None
        Hessian, shape ``(n_dim, n_dim)``, as differentiated (not symmetrized); None when
        not requested.
    finite : jax.Array
        Scalar bool, True when every requested quantity is finite everywhere. A caller
        decides what a False means for it; the helper never substitutes another matrix.
    """

    value: Any
    grad: Any
    hess: Any
    finite: Any


#: Compiled programs keyed weakly on the function, then on the requested quantities. A
#: program holds its function only through a weak reference, so a dropped function frees
#: its entry, and repeated calls on one function compile once.
_PROGRAMS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _evaluate(
    fn: Callable[..., jax.Array], x: Any, args: tuple, want: tuple[str, ...]
) -> Curvature:
    """Differentiate ``fn(x, *args)``; the traced body shared by every call."""
    x = jnp.asarray(x)
    value = grad = hess = None
    if "value" in want and "grad" in want:
        value, grad = jax.value_and_grad(fn)(x, *args)
    elif "value" in want:
        value = fn(x, *args)
    elif "grad" in want:
        grad = jax.grad(fn)(x, *args)
    if "hess" in want:
        hess = jax.hessian(fn)(x, *args)
    finite = jnp.asarray(True)
    for part in (value, grad, hess):
        if part is not None:
            finite = finite & jnp.all(jnp.isfinite(part))
    return Curvature(value, grad, hess, finite)


def _program(fn: Callable[..., jax.Array], want: tuple[str, ...]):
    """The jitted derivative program of ``fn`` for ``want``, built once per function."""
    try:
        per_fn = _PROGRAMS.get(fn)
        if per_fn is None:
            per_fn = _PROGRAMS[fn] = {}
    except TypeError:  # not weak-referenceable: compile without caching
        return jax.jit(lambda x, *args: _evaluate(fn, x, args, want))
    program = per_fn.get(want)
    if program is None:
        fn_ref = weakref.ref(fn)

        def live(x, *args):
            f = fn_ref()
            if f is None:
                raise RuntimeError("the function was collected while its derivatives were in use")
            return _evaluate(f, x, args, want)

        program = per_fn[want] = jax.jit(live)
    return program


def curvature(
    fn: Callable[..., jax.Array],
    x: jax.Array,
    *args: Any,
    want: tuple[str, ...] = ("grad", "hess"),
) -> Curvature:
    """Differentiate ``fn(x, *args)`` once or twice at ``x``.

    Parameters
    ----------
    fn : callable
        ``fn(x, *args) -> scalar``, JAX-traceable. Only ``x`` is differentiated. Pass the
        same function object on repeated calls to reuse the compiled program.
    x : array_like, shape (n_dim,)
        Point at which to evaluate.
    *args : pytree
        Passed through to ``fn``; not differentiated.
    want : tuple of {"value", "grad", "hess"}, optional
        Quantities to compute. Default ``("grad", "hess")``.

    Returns
    -------
    Curvature
        The requested quantities and their joint finiteness.

    Notes
    -----
    JIT/vmap-safe (inside a trace the program is inlined). Called eagerly it is one
    compiled program per ``(fn, want, shapes, dtype)``, never an op-by-op dispatch of the
    model. The Hessian is :func:`jax.hessian` (forward over reverse), the only call of it in
    the inference package. A non-finite entry is reported in ``finite``, never replaced.
    """
    unknown = set(want) - _FIELDS
    if unknown:
        raise ValueError(
            f"unknown curvature quantity {sorted(unknown)}; choose from {sorted(_FIELDS)}"
        )
    return _program(fn, tuple(want))(x, *args)
