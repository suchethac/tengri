# SPDX-License-Identifier: BSD-3-Clause
"""One way to assert that ``jax.jit`` did not change the answer.

Why this exists
---------------
A census of the suite found 304 tests with ``jit`` in the name. 70 of the 89
named ``test_jit_compatible`` asserted only that the compiled call returned
*finite* numbers::

    jitted = jax.jit(f)
    out = jitted(x)
    chex.assert_tree_all_finite(out)  # passes for any finite garbage

That is a compile-doesn't-crash smoke test wearing the name of a correctness
test. It cannot fail for the bug it is named after: if ``jit`` returns wrong
values — a Python ``int`` captured as a trace-time constant, a branch taken on
a tracer's truthiness, a silent ``float64 -> float32`` demotion — every element
is still finite and the assertion is still green.

The property worth pinning is *parity*: compiled output equals eager output.
Roughly 50 tests in this tree already did that by hand, under four different
names (``test_jit_parity_vs_eager``, ``test_jit_matches_eager``,
``test_jit_parity``, ``test_jit_parity_and_shape``). This module is that check,
once, so the remaining sites can adopt it instead of reinventing a fifth
spelling.

Tolerance
---------
``rtol=1e-6`` by default, matching the existing per-component guard in
``tests/integration/test_retrace_guards.py``. JIT/eager parity is *near*-exact
rather than exact: XLA fuses and reorders floating-point operations, so results
differ in the last bits — ~1e-15 for float64, ~1e-7 for float32. Asserting
exact equality would flake on fusion changes; asserting nothing is the failure
this module exists to correct.

Callers that know their kernel is bit-reproducible should tighten it
(``rtol=1e-12`` is used by the WG00 geometry tests and holds there). A test
that needs a tolerance looser than ~1e-5 is reporting a real numerical
sensitivity and should say so in a comment rather than widen silently.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")

#: Absolute floor as a fraction of the output's own peak magnitude. Differences
#: below this are numerical noise (denormal flush-to-zero, fused-multiply-add
#: reassociation), not physics. See ``_noise_floor``.
_PEAK_RELATIVE_FLOOR = 1e-12


def _noise_floor(tree: Any) -> float:
    """An absolute tolerance scaled to the magnitude of ``tree`` itself.

    A pure relative tolerance is undefined where the reference value is exactly
    zero, and SEDs are zero by construction outside their emission range. The
    first ``test_jit_compatible`` converted to a parity check tripped on exactly
    that: ``compute_nlr_sed_richardson2014`` returned ``1.8e-275`` under ``jit``
    and a hard ``0.0`` eagerly at 1 of 200 wavelengths — a denormal that one
    path flushed and the other did not. ``|1.8e-275 - 0| / 0`` is ``inf``, so a
    relative-only comparison reports an infinite error over a difference 275
    orders of magnitude below anything observable.

    Scaling the floor to the peak — rather than hard-coding an absolute number
    — keeps the check meaningful across this codebase's range, where an SED may
    peak at 1e-30 or at 1e44 depending on units and luminosity. A genuine JIT
    defect (a constant folded at trace time, a branch taken on a tracer, an
    f64->f32 demotion) shifts values by O(1) *relative* error, which is twelve
    orders of magnitude above this floor and still caught.
    """
    peak = 0.0
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(leaf)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.floating):
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            peak = max(peak, float(np.max(np.abs(finite))))
    return _PEAK_RELATIVE_FLOOR * peak


def assert_jit_matches_eager(
    fn: Callable[..., T],
    *args: Any,
    rtol: float = 1e-6,
    atol: float | None = None,
    **kwargs: Any,
) -> T:
    """Call ``fn`` eagerly and under ``jax.jit``; assert the results agree.

    Returns the *jitted* result, so an existing test can swap
    ``jax.jit(f)(x)`` for ``assert_jit_matches_eager(f, x)`` and keep every
    assertion it already made — the swap can only add coverage.

    Args:
        fn: the callable to compile. Passed to ``jax.jit`` as-is, so it must
            be traceable with the given arguments (which the pre-existing
            ``jax.jit(fn)(...)`` call already established).
        *args, **kwargs: forwarded to both the eager and the compiled call.
        rtol: relative tolerance for the comparison. See the module docstring.
        atol: absolute tolerance. ``None`` (the default) derives one from the
            eager output's peak magnitude — see ``_noise_floor``. Pass ``0.0``
            to demand exact agreement wherever the reference is zero.

    Raises:
        AssertionError: if compiled and eager results differ beyond tolerance,
            with chex's per-leaf diff naming the offending pytree path.
    """
    eager = fn(*args, **kwargs)
    jitted = jax.jit(fn)(*args, **kwargs)
    tol = _noise_floor(eager) if atol is None else atol
    chex.assert_trees_all_close(jitted, eager, rtol=rtol, atol=tol)
    return jitted


def assert_vmap_matches_loop(
    fn: Callable[..., Any],
    batched_arg: Any,
    *,
    rtol: float = 1e-6,
    atol: float | None = None,
    in_axes: Any = 0,
) -> Any:
    """Assert ``vmap(fn)`` over ``batched_arg`` equals looping ``fn`` per row.

    The vmap counterpart to :func:`assert_jit_matches_eager`, and it exists for
    the same reason: of 81 tests in this tree that call ``vmap``, only 31
    compare the batched result against anything. ``vmap`` that silently
    broadcasts along the wrong axis still returns a finite array of the
    expected shape — the looped reference is what distinguishes "batched" from
    "batched correctly".
    """
    stacked = jax.vmap(fn, in_axes=in_axes)(batched_arg)
    looped = jnp.stack([fn(row) for row in batched_arg])
    tol = _noise_floor(looped) if atol is None else atol
    chex.assert_trees_all_close(stacked, looped, rtol=rtol, atol=tol)
    return stacked


def assert_jit_matches_eager_static(
    fn: Callable[..., T],
    *args: Any,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: str | tuple[str, ...] | None = None,
    rtol: float = 1e-6,
    atol: float | None = None,
    **kwargs: Any,
) -> T:
    """``assert_jit_matches_eager`` for kernels that need static arguments.

    Separate from the common path so the common path stays a two-argument
    call. A string mode flag or a shape-determining ``int`` must be static or
    tracing fails outright — that failure is not the bug this check hunts.
    """
    jit_kwargs: dict[str, Any] = {}
    if static_argnums is not None:
        jit_kwargs["static_argnums"] = static_argnums
    if static_argnames is not None:
        jit_kwargs["static_argnames"] = static_argnames

    eager = fn(*args, **kwargs)
    jitted = jax.jit(fn, **jit_kwargs)(*args, **kwargs)
    tol = _noise_floor(eager) if atol is None else atol
    chex.assert_trees_all_close(jitted, eager, rtol=rtol, atol=tol)
    return jitted
