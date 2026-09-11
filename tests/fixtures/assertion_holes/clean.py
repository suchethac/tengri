# SPDX-License-Identifier: BSD-3-Clause
"""Assertions that satisfy the finite-AND-non-zero rule, one per accepted form.

The companion to ``holes.py``: a guard that flags everything is as useless as one
that flags nothing, so the contract test pins that none of these is reported.
"""

import jax
import numpy as np


def paired_in_two_assertions(f, x):
    """The canonical form. Order is not the point; having both is."""
    g = jax.grad(f)(x)
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


def paired_in_one_assertion(f, x):
    """The same claim spelled as one conjunction."""
    g = jax.grad(f)(x)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def a_lower_bound_settles_both(f, x):
    """``> 0`` is not a half-claim. Every ordered comparison with NaN is ``False``
    and zero fails the bound, so a lower bound names the good state on its own."""
    g = jax.grad(f)(x)
    assert float(np.max(np.abs(g))) > 0.0


def zero_is_the_subject_not_the_accident(f, x):
    """A gradient pinned *as* zero has no good state to name — the bad state is
    what the test is about, and demanding a non-zero partner would be nonsense."""
    g = jax.grad(f)(x)
    assert np.all(g == 0.0)


def suppressed_by_a_marker_above_the_assert(f, x):
    g = jax.grad(f)(x)
    # grad-assert: finite-only — this band is masked, a zero gradient is the point
    assert np.all(np.isfinite(g))


def suppressed_by_a_marker_on_the_assert(f, x):
    g = jax.grad(f)(x)
    assert np.any(g != 0.0)  # grad-assert: nonzero-only — finiteness is pinned upstream
