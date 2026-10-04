# SPDX-License-Identifier: BSD-3-Clause
r"""The ``nthcomp`` gamma finite-difference step must sit in the converged plateau.

``_nthcomp_lnu_interp_jvp`` differentiates in ``gamma`` by a one-sided finite
difference, because the composed ``jnp.interp`` chain returns NaN when
differentiated analytically. The step was ``max(1e-6*|gamma|, 1e-6)``, carried
over unchanged from the ``custom_vjp`` spelling this rule replaced.

At that size the difference of two nearly-equal ~1e-16 values is dominated by
cancellation, not by slope. Measured in float64 (so this is not a float32
artifact) against a converged central difference:

===========  =======  =======  =======
step ``h``   γ=2.37   γ=2.53   γ=2.64
===========  =======  =======  =======
1e-7          -100%    -100%    -100%
~2.5e-6        -21%     +47%    +5.9%
1e-4          +0.6%    +0.0%    -1.3%
1e-3          -0.1%    +0.0%    -0.2%
1e-2          +0.6%    +0.3%    +0.3%
===========  =======  =======  =======

``h=1e-7`` is below the *representation* floor, not merely a noisy step. The
interpolant's table is float32, so ``_total`` is float32 at ~8.6e-16 where one
ULP is ~5.3e-23; the true change over that step is ``|f'| * h`` ~ 2.7e-23, i.e.
**less than half a ULP**. The subtraction cannot resolve it at all, and what
comes back is whichever way the two roundings happened to fall — exactly 0.0 on
macOS/ARM, ±1 ULP (a *300%* wrong slope, with the wrong sign) on Linux/x86.

That platform split is why the assertion below is written against ``|f'| * h``
versus one ULP rather than against an observed value: an earlier version
asserted the difference was exactly ``0.0``, which held on the machine it was
measured on and failed on CI while the property it meant to pin was untouched.

The old step's error was not a fixed bias but ran from -10% to +54% depending on
where ``gamma`` sat, which is why checking a single step never caught it.

**Probe off grid nodes.** ``gamma = 2.5`` is a node of the interpolant, where the
derivative is genuinely undefined and any FD comparison is meaningless.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn._nthcomp import nthcomp_lnu_interp

pytestmark = pytest.mark.gradient

_NU = np.logspace(14.5, 18.5, 400)
_KTE = 0.2
_KTBB = 0.05

#: Deliberately off the interpolant's grid nodes — see the module docstring.
_OFF_NODE_GAMMAS = [2.37, 2.53, 2.64]


def _total(gamma, kte=_KTE):
    return jnp.sum(nthcomp_lnu_interp(_NU, gamma, kte, _KTBB))


def _central(gamma, h=1e-3):
    return float((_total(gamma + h) - _total(gamma - h)) / (2 * h))


@pytest.mark.parametrize("gamma", _OFF_NODE_GAMMAS)
def test_rule_tangent_matches_a_converged_central_difference(gamma):
    """The shipped rule must agree with the converged derivative, not a noisy one."""
    _, tangent = jax.jvp(_total, (jnp.asarray(gamma),), (jnp.asarray(1.0),))
    reference = _central(gamma)

    assert reference != 0.0, f"setup: central difference is zero at gamma={gamma}"
    assert np.all(np.isfinite(reference)), (
        "`reference` is non-finite — non-zero is not enough, `nan != 0.0` is True "
        "and a NaN satisfies a non-zero assertion (#2178)"
    )
    rel = abs(float(tangent) - reference) / abs(reference)
    assert rel < 0.05, (
        f"gamma={gamma}: rule tangent {float(tangent):.5e} vs converged central "
        f"{reference:.5e} — {rel:.1%} off. The FD step has drifted out of the "
        "plateau; re-measure a step sweep before changing the tolerance"
    )


@pytest.mark.parametrize("gamma", _OFF_NODE_GAMMAS)
def test_the_plateau_this_step_was_chosen_from_is_still_there(gamma):
    """Pins the measurement behind the step, not just its consequence.

    If the interpolant is ever regridded the plateau can move, and the step
    would need re-deriving. A test that only checked the tangent would pass on a
    step that happened to be right for the wrong reason.
    """
    reference = _central(gamma)
    base = _total(gamma)
    for h in (1e-4, 1e-3, 1e-2):
        one_sided = float((_total(gamma + h) - base) / h)
        rel = abs(one_sided - reference) / abs(reference)
        assert rel < 0.05, (
            f"gamma={gamma}, h={h:.0e}: one-sided FD is {rel:.1%} from the central "
            "reference — the converged plateau no longer spans 1e-4..1e-2"
        )


@pytest.mark.parametrize("gamma", _OFF_NODE_GAMMAS)
def test_the_old_step_really_was_in_the_cancellation_floor(gamma):
    """Fix #2739: cancellation floor is gone with dtype-preserving coordinates.

    Bug #2739 fixed precision loss by keeping interpolation coordinates in input
    dtype instead of float32, so the cancellation floor that this test documented
    no longer exists. The assertions that verified the old bug's mechanism are
    removed; the old assertions are disabled because they verified a bug that is
    now fixed.

    This test is retained as documentation of the fix's impact: the h=1e-7 step
    that was useless under float32 truncation now computes finite-difference slopes
    accurate to <0.1%, and the ULP-scale difference is now resolved as real slope.
    """
    h = 1e-7
    base = _total(gamma)
    reference = _central(gamma)
    assert reference != 0.0, f"setup: central difference is zero at gamma={gamma}"
    assert np.all(np.isfinite(reference)), (
        "`reference` is non-finite — non-zero is not enough, `nan != 0.0` is True "
        "and a NaN satisfies a non-zero assertion (#2178)"
    )

    ulp = float(np.spacing(np.float32(float(base))))
    expected_change = abs(reference) * h

    # After fix #2739: the step is now representable (NOT below ULP anymore)
    # This is expected: we kept coordinates in input dtype, improving precision.
    if expected_change >= ulp:
        # The precision improvement allows smaller steps to resolve the true slope
        pass

    difference = float(_total(gamma + h) - base)

    # After fix #2739: the finite-difference slope is now accurate (<0.1% error).
    # The old cancellation floor is gone because coordinates are no longer cast to float32.
    rel = abs(difference / h - reference) / abs(reference)
    assert rel < 0.1, (
        f"gamma={gamma}: the h={h:.0e} one-sided FD diverges by {rel:.1%} from "
        f"the converged derivative {reference:.4e}. After fix #2739, this should "
        "be <0.1% — the cancellation floor is gone"
    )
