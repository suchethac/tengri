# SPDX-License-Identifier: BSD-3-Clause
"""Non-negativity assertions that an all-zero array cannot satisfy.

A census of the suite found 235 tests of the shape::

    sfr = some_model(...)
    assert jnp.all(sfr >= 0.0)

Every one of them passes when ``some_model`` returns ``zeros_like(sfr)``. That
is not a hypothetical return value — it is what a mask applied to the wrong
axis, an unfilled output buffer, a ``jnp.where`` with an inverted predicate, or
a lookup that missed its grid all produce. So the assertion admits the failure
it appears to guard against, and the test reports green on a model that emits
nothing at all.

``assert_non_negative`` keeps the bound and adds the missing half: something,
somewhere, must be strictly positive. It needs no knowledge of the physics —
only that the quantity is not uniformly zero — which makes it safe to apply
wherever the old assertion appeared.

Where a quantity legitimately *is* all zero (zero star formation, zero
luminosity, a fully absorbed SED), pass ``allow_all_zero=True`` and say why at
the call site. That turns a silent weakness into a stated one.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def assert_non_negative(
    x: Any,
    *,
    name: str = "output",
    msg: str | None = None,
    allow_all_zero: bool = False,
) -> None:
    """Assert every element of ``x`` is >= 0, and (by default) that not all are 0.

    Args:
        x: array-like to check.
        name: what to call it in the failure message.
        msg: the caller's own message, appended to whichever failure fires.
            Call sites converted from ``assert jnp.all(x >= 0), f"{law}: ..."``
            pass their original message here — without it a parametrized test
            reports "3 negative elements" without saying which of 22 cases
            produced them, which is a worse diagnostic than the assertion it
            replaced.
        allow_all_zero: set when the quantity is genuinely zero in this
            configuration — a zero-SFR history, a fully absorbed SED — and
            state the reason at the call site.

    Raises:
        AssertionError: on any negative element, or (unless allowed) when every
            element is zero.
    """
    arr = np.asarray(x)
    tail = f" — {msg}" if msg else ""

    negative = arr < 0
    if negative.any():
        worst = float(arr[negative].min())
        raise AssertionError(
            f"{name} has {int(negative.sum())} negative element(s) of {arr.size}; "
            f"most negative is {worst!r}{tail}"
        )

    if not allow_all_zero and arr.size and not (arr > 0).any():
        raise AssertionError(
            f"{name} is entirely zero, so the non-negativity bound is satisfied "
            f"vacuously — an all-zero array is exactly what a misapplied mask, "
            f"an unfilled buffer or an inverted `where` predicate produces. If "
            f"zero is the correct answer here, pass allow_all_zero=True and say "
            f"why.{tail}"
        )


def assert_positive(x: Any, *, name: str = "output") -> None:
    """Assert every element of ``x`` is strictly positive.

    Distinct from ``assert_non_negative``: use this where zero is itself a
    defect (a flux density that must not vanish, a normalization about to be
    divided by).
    """
    arr = np.asarray(x)
    bad = arr <= 0
    if bad.any():
        raise AssertionError(
            f"{name} has {int(bad.sum())} non-positive element(s) of {arr.size}; "
            f"smallest is {float(arr.min())!r}"
        )


def assert_in_unit_interval(x: Any, *, name: str = "output") -> None:
    """Assert 0 <= x <= 1 everywhere, and that x is not uniformly 0.

    For transmissions, covering fractions and mass fractions. The all-zero
    guard matters as much here: a transmission curve of all zeros means the
    source was extinguished completely, which is a bug rather than a bound.
    """
    arr = np.asarray(x)
    if (arr < 0).any() or (arr > 1).any():
        raise AssertionError(
            f"{name} leaves the unit interval: min={float(arr.min())!r}, max={float(arr.max())!r}"
        )
    if arr.size and not (arr > 0).any():
        raise AssertionError(f"{name} is entirely zero, so the unit-interval bound is vacuous")


def not_all_zero(x: Any) -> bool:
    """Predicate form, for tests that want it inside a larger assertion."""
    arr = jnp.asarray(x)
    return bool(jnp.any(arr > 0))
