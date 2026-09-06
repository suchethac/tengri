# SPDX-License-Identifier: BSD-3-Clause
"""One of each hole shape the finite-AND-non-zero rule exists to catch.

Not a test module and not collected: every function here is a **mutation**, a
deliberate reproduction of a hole this repository has already shipped, kept
executable so ``tests/contract/test_gradient_assertion_guard.py`` can prove
``tools/check_gradient_assertions.py`` still detects it. A guard that cannot
demonstrate it catches the bugs that motivated it is not a guard.

``tests/fixtures/`` is skipped by the guard's directory sweep, so these holes
cannot fail CI merely by existing. The contract test points the guard at this
file explicitly.
"""

import jax
import jax.numpy as jnp
import numpy as np


def hole_2100_finite_only(f, x):
    """#2100: a gradient pinned *finite*. Zero is finite, so an identically zero
    gradient satisfies this and the test goes green on a dead objective."""
    g = jax.grad(f)(x)
    assert np.all(np.isfinite(g))


def hole_2178_nonzero_only(f, x):
    """#2178: the mirror. A gradient pinned *non-zero*. ``nan != 0.0`` is ``True``,
    so a NaN satisfies this and reads as a repaired underflow."""
    g = jax.grad(f)(x)
    assert np.any(g != 0.0)


def hole_2100_through_a_pytree(f, x):
    """The #2100 shape with the gradient renamed on the way: ``leaves`` is still
    ``g``, which is why this guard tracks taint through assignment rather than
    matching names in the source text."""
    g = jax.grad(f)(x)
    leaves = [np.asarray(v) for v in jax.tree_util.tree_leaves(g)]
    assert all(np.all(np.isfinite(v)) for v in leaves)


def hole_2178_on_a_scalar(f, x):
    """The #2178 shape written as a scalar comparison. ``float('nan') != 0.0`` is
    ``True`` exactly as the array form is."""
    grad = float(jax.grad(f)(x))
    assert grad != 0.0


def hole_named_but_not_traced(x):
    """No ``jax.grad`` in sight — the name ``g32`` is what declares this a gradient,
    and the #2100 seam checks were written in exactly this style."""
    g32 = jnp.asarray(x, dtype=jnp.float32)
    assert jnp.all(jnp.isfinite(g32))


def hole_behind_a_negated_isnan(f, x):
    """``not isnan`` is a finiteness claim written the other way round, and it
    admits zero just as ``isfinite`` does."""
    g = jax.grad(f)(x)
    assert not np.any(np.isnan(g))


def hole_marker_without_a_reason(f, x):
    """An escape hatch that costs no explanation is a mute button, not a hatch."""
    g = jax.grad(f)(x)
    assert np.all(np.isfinite(g))  # grad-assert: finite-only


def hole_marker_without_a_half(f, x):
    """A marker that will not say which half it skips suppresses nothing."""
    g = jax.grad(f)(x)
    assert np.all(np.isfinite(g))  # grad-assert: it is fine, trust me


def precision_hole_on_a_forward_value(model, params):
    """Not a gradient at all. In ``tests/regression/precision/`` the question is
    whether a number survives float32, and a forward that has collapsed to zero
    fails that question exactly as a NaN does — #2178's own fix had to add
    ``assert np.any(fluxes != 0.0)`` beside a finite check for this reason."""
    flux = model.predict_photometry(params)
    assert np.all(np.isfinite(flux))
