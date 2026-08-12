# SPDX-License-Identifier: BSD-3-Clause
"""Compare ``jax.grad`` against a finite difference, cheaply.

Why this exists
---------------
The census that produced ``tests/_jit_parity.py`` found the same defect in the
gradient tests, at four times the scale. Of 367 tests that call ``jax.grad``,
165 compare against a numerical derivative and **202 assert only that the
gradient is finite**::

    g = jax.grad(total_sfr)(0.3)
    assert jnp.isfinite(g)

A gradient that is wrong — wrong sign, wrong by a factor of the bin width, zero
because the parameter was accidentally detached by ``stop_gradient`` or an
integer cast — is still finite. The assertion cannot fail for the bug it is
named after. Since every fitter in this codebase (MAP, VI, NUTS) descends these
gradients, a silently wrong one does not crash; it converges somewhere wrong.

Cost
----
The check is two extra function evaluations, *independent of dimension*. For an
array input it does not build the Jacobian: it draws one fixed random unit
vector ``v`` and compares the directional derivative ``<grad f, v>`` against
``(f(x + eps v) - f(x - eps v)) / 2 eps``. A wrong gradient is almost surely
wrong along a random direction, so this catches essentially what a full
Jacobian comparison would, at O(1) instead of O(n) evaluations. ``v`` is drawn
from a fixed seed so a failure reproduces exactly.

When NOT to use it
------------------
Finite differences are only meaningful where the function is smooth at ``x``.
This codebase deliberately contains non-smooth kernels — the non-parametric
SFHs use ``searchsorted`` to produce piecewise-constant step functions, and
there is a regression test asserting they stay that way. At a bin boundary the
analytic derivative and the finite difference legitimately disagree, and a
failure there means this check is the wrong tool, not that the gradient is
wrong. Keep a finiteness assertion at those sites and say so in a comment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

#: Relative step for the central difference. Central differences carry O(eps^2)
#: truncation error and O(macheps/eps) cancellation error; with float64 enabled
#: (tests/conftest.py does this globally) 1e-5 sits near the minimum of the sum.
_REL_STEP = 1e-5

#: Seed for the probe direction. Fixed so a failure is reproducible rather than
#: appearing on one run in ten.
_PROBE_SEED = 20260812


def _step_for(x: Any) -> float:
    """A step scaled to the largest magnitude anywhere in ``x``, never zero."""
    scale = 0.0
    for leaf in jax.tree_util.tree_leaves(x):
        arr = jnp.asarray(leaf)
        if jnp.size(arr):
            scale = max(scale, float(jnp.max(jnp.abs(arr))))
    return _REL_STEP * (scale if scale > 0.0 else 1.0)


def _random_unit_tangent(x: Any) -> Any:
    """A random unit-norm tangent with the same pytree structure as ``x``.

    Probing along one random direction costs two function evaluations however
    many parameters there are; a full Jacobian comparison costs 2N. A gradient
    that is wrong is wrong along a random direction with probability 1, so the
    cheap probe loses almost nothing.
    """
    leaves, treedef = jax.tree_util.tree_flatten(x)
    keys = jax.random.split(jax.random.PRNGKey(_PROBE_SEED), max(len(leaves), 1))
    tangents = [
        jax.random.normal(k, jnp.shape(jnp.asarray(leaf)), dtype=jnp.asarray(leaf).dtype)
        for k, leaf in zip(keys, leaves)
    ]
    norm = jnp.sqrt(sum(jnp.sum(t**2) for t in tangents))
    norm = jnp.where(norm == 0.0, 1.0, norm)
    return jax.tree_util.tree_unflatten(treedef, [t / norm for t in tangents])


def _dot(a: Any, b: Any) -> float:
    return float(
        sum(
            jnp.sum(jnp.asarray(x) * jnp.asarray(y))
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
        )
    )


def _shift(x: Any, v: Any, step: float) -> Any:
    return jax.tree_util.tree_map(lambda a, b: a + step * b, x, v)


def _fd_noise_floor(f_plus: Any, f_minus: Any, step: float) -> float:
    """The smallest derivative a central difference can distinguish from zero.

    Differencing amplifies the function's *own* rounding error by 1/(2h): if f
    is only known to within one rounding quantum ``|f| * eps_machine``, the
    quotient inherits ``|f| * eps_machine / h`` of noise no matter how exact
    the analytic gradient is.

    Ignoring this produces a false positive on every legitimately-zero gradient
    computed in single precision, and one did:
    ``test_met_table_grad_wrt_lgmet`` sums float32 age weights to |f| ~ 4 and
    has an exactly-zero derivative w.r.t. the metallicity table. With h = 2e-5
    the floor is 4 * 1.19e-7 / 2e-5 = 2.4e-2 — and the probe duly reported
    -2.384e-2, one float32 quantum of f divided by 2h, against an analytic 0.0.
    The gradient was right; the reference was noise.

    The factor of 2 covers the two independent roundings that are subtracted.
    """
    mag = max(abs(float(f_plus)), abs(float(f_minus)))
    dtype = np.asarray(f_plus).dtype
    resolution = float(np.finfo(dtype).eps) if np.issubdtype(dtype, np.floating) else 0.0
    return 2.0 * mag * resolution / abs(step) if step else 0.0


def assert_grad_matches_fd(
    f: Callable[[Any], Any],
    x: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-8,
    eps: float | None = None,
) -> Any:
    """Assert ``jax.grad(f)(x)`` agrees with a central finite difference.

    Returns the analytic gradient, so a converted test keeps whatever
    assertions it already made — the swap can only add coverage.

    Args:
        f: scalar-valued function of a single argument.
        x: point at which to differentiate. A scalar, an array, or any pytree
            of arrays — a dict of named parameters is the usual shape here, and
            it is probed along a single random tangent of the same structure.
        rtol: relative tolerance. The default is loose by design — a central
            difference is only ~1e-10 accurate in the best case, and the errors
            worth catching (sign flips, missing chain-rule factors, gradients
            silently zeroed) are O(1) relative.
        atol: absolute floor, for points where the true derivative is ~0 and a
            relative comparison is meaningless.
        eps: override the step. Supply this where the function has structure on
            a scale finer than ``1e-5 * |x|``.

    Raises:
        AssertionError: if the analytic and numerical derivatives disagree.
    """
    is_leaf_scalar = not isinstance(x, (dict, list, tuple)) and jnp.ndim(x) == 0

    if is_leaf_scalar:
        point: Any = jnp.asarray(x, dtype=jnp.result_type(float))
        step = _step_for(point) if eps is None else eps
        grad = jax.grad(f)(point)
        analytic = float(grad)
        f_plus, f_minus = f(point + step), f(point - step)
        direction = "d/dx"
    else:
        point = x
        step = _step_for(point) if eps is None else eps
        grad = jax.grad(f)(point)
        v = _random_unit_tangent(point)
        analytic = _dot(grad, v)
        f_plus, f_minus = f(_shift(point, v, step)), f(_shift(point, v, -step))
        direction = "directional derivative along a fixed random unit tangent"

    numeric = float((f_plus - f_minus) / (2.0 * step))

    if not np.isfinite(numeric):
        raise AssertionError(
            f"the finite-difference probe itself was not finite ({numeric}); "
            f"f is not evaluable at x +/- {step:g}, so this check cannot judge "
            f"the gradient here."
        )

    if float(f_plus) == float(f_minus) and abs(analytic) > atol:
        raise AssertionError(
            f"the finite-difference probe could not resolve any change in f: "
            f"f(x + h) and f(x - h) are bit-identical at h={step:g}, so the "
            f"numerical derivative underflows to 0 while jax.grad reports "
            f"{analytic!r}. This happens when |x| is so small that a step "
            f"proportional to it is lost in f's own rounding — it is a limit "
            f"of the probe, not evidence about the gradient. Pass an explicit "
            f"eps, or keep a finiteness assertion at this site."
        )

    tol = max(atol, _fd_noise_floor(f_plus, f_minus, step))

    np.testing.assert_allclose(
        analytic,
        numeric,
        rtol=rtol,
        atol=tol,
        err_msg=(
            f"analytic gradient disagrees with a central finite difference "
            f"({direction}, eps={step:g}, fd noise floor {tol:g}). "
            f"analytic={analytic!r} "
            f"numeric={numeric!r}. If this function is non-smooth at x — a "
            f"searchsorted bin edge, a clip, an abs — the finite difference is "
            f"not a valid reference and this check is the wrong tool here."
        ),
    )
    return grad
