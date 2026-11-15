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
import numpy as np

__all__ = ["Curvature", "NonFiniteCurvatureError", "curvature", "flat_names", "require_finite"]

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

    def require_finite(self, *, names=None, where: str = "curvature") -> Curvature:
        """Raise :class:`NonFiniteCurvatureError` unless every requested quantity is finite.

        Parameters
        ----------
        names : sequence of str or callable, optional
            Names of the flat coordinates, or a callable returning them (called only
            when something is non-finite).
        where : str, optional
            Label of the consumer, first words of the message.

        Returns
        -------
        Curvature
            ``self``, so ``curvature(...).require_finite()`` reads as one expression.

        Notes
        -----
        Eager only: inside a trace the flag is a tracer, the check is skipped, and the
        caller reads ``finite`` itself.
        """
        require_finite(value=self.value, grad=self.grad, hess=self.hess, names=names, where=where)
        return self


class NonFiniteCurvatureError(ValueError):
    """A value, gradient or Hessian is NaN or inf at the point it was evaluated."""


def flat_names(unravel: Callable[[Any], Any], n_dim: int) -> tuple[str, ...]:
    """Names of the coordinates of a flat vector, from the pytree its ``unravel`` rebuilds.

    Parameters
    ----------
    unravel : callable
        Inverse of a ``jax.flatten_util.ravel_pytree`` flattening: ``(n_dim,) -> pytree``.
    n_dim : int
        Length of the flat vector.

    Returns
    -------
    tuple of str
        One name per coordinate: the leaf's key path, indexed (``psd_xi[3]``) when the
        leaf holds more than one coordinate. A coordinate the pytree does not expose is
        named by its index.
    """
    tree = unravel(jnp.arange(n_dim, dtype=jnp.float32))
    names = [f"[{i}]" for i in range(n_dim)]
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        label = jax.tree_util.keystr(path).strip("[]'\"") or "x"
        flat = np.asarray(leaf).ravel()
        for j, index in enumerate(flat):
            names[int(index)] = label if flat.size == 1 else f"{label}[{j}]"
    return tuple(names)


def _describe(label: str, array: np.ndarray, names) -> str | None:
    bad = ~np.isfinite(array)
    if not bad.any():
        return None
    if array.ndim == 0:
        return f"{label} is {array}"
    index = np.nonzero(bad.any(axis=1) if array.ndim == 2 else bad)[0]
    if names is not None and callable(names):
        names = names()
    shown = ", ".join(
        str(names[i]) if names is not None and i < len(names) else str(int(i)) for i in index[:8]
    )
    more = f" and {index.size - 8} more" if index.size > 8 else ""
    axis = "rows" if array.ndim == 2 else "entries"
    return (
        f"{label} has {int(bad.sum())} of {array.size} entries non-finite ({axis} {shown}{more})"
    )


def require_finite(*, value=None, grad=None, hess=None, names=None, where: str = "curvature"):
    """Raise unless the value, gradient and Hessian given are all finite.

    The one rule every curvature consumer applies before using a Hessian: a non-finite
    matrix is a fact about the objective at this point, in float64 as in float32, and it
    is reported by name rather than passed on to an eigensolver that would return NaN.

    Parameters
    ----------
    value, grad, hess : array_like or None
        The quantities to check (scalar, ``(n_dim,)``, ``(n_dim, n_dim)``).
    names : sequence of str or callable, optional
        Coordinate names, or a callable returning them (called only on failure).
    where : str, optional
        Consumer label for the message.

    Raises
    ------
    NonFiniteCurvatureError
        Naming which of value, gradient and Hessian is non-finite, the rows or entries
        involved, and the dtype.

    Notes
    -----
    Eager only: a traced argument skips the check (the caller reads
    :attr:`Curvature.finite`).
    """
    parts = (("value", value), ("gradient", grad), ("Hessian", hess))
    if any(isinstance(a, jax.core.Tracer) for _, a in parts):
        return
    problems, dtype = [], None
    for label, a in parts:
        if a is None:
            continue
        array = np.asarray(a)
        dtype = dtype or str(array.dtype)
        found = _describe(label, array, names)
        if found:
            problems.append(found)
    if problems:
        raise NonFiniteCurvatureError(
            f"{where}: {'; '.join(problems)} (dtype {dtype}). The objective's derivative is "
            "not finite at this point, whatever the dtype: a likelihood, prior or model "
            "term returns NaN or inf there. Probe it column by column with "
            "jax.jvp(jax.grad(f), ...) to find the term."
        )


#: Compiled programs keyed on the identity of the function (never its ``__eq__`` or
#: ``__hash__``, which a callable may define by value), then on the requested quantities.
#: An entry holds its function only through a weak reference and removes itself when the
#: function is collected, so a recycled ``id`` cannot reach a stale program.
_PROGRAMS: dict[int, tuple[weakref.ref, dict]] = {}


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
    """The jitted derivative program of ``fn`` for ``want``, built once per function object."""
    key = id(fn)
    entry = _PROGRAMS.get(key)
    if entry is not None and entry[0]() is not fn:
        entry = None
    if entry is None:

        def drop(ref, key=key):
            current = _PROGRAMS.get(key)
            if current is not None and current[0] is ref:
                del _PROGRAMS[key]

        try:
            ref = weakref.ref(fn, drop)
        except TypeError:  # not weak-referenceable: compile without caching
            return jax.jit(lambda x, *args: _evaluate(fn, x, args, want))
        entry = _PROGRAMS[key] = (ref, {})
    programs = entry[1]
    program = programs.get(want)
    if program is None:
        fn_ref = entry[0]

        def live(x, *args):
            f = fn_ref()
            if f is None:
                raise RuntimeError("the function was collected while its derivatives were in use")
            return _evaluate(f, x, args, want)

        program = programs[want] = jax.jit(live)
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
        same function object on repeated calls to reuse the compiled program (the cache
        is keyed on object identity); a fresh closure or bound method compiles on every call.
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
    The helper declines no dtype: float32 and float64 take the same path, and every
    consumer reads the result through ``finite``.
    """
    unknown = set(want) - _FIELDS
    if unknown:
        raise ValueError(
            f"unknown curvature quantity {sorted(unknown)}; choose from {sorted(_FIELDS)}"
        )
    return _program(fn, tuple(want))(x, *args)
