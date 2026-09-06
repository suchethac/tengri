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

_NU = jnp.asarray(np.logspace(14.5, 18.5, 400))
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
    """The regression this guards against, stated as a measurement.

    Documents *why* 1e-6 was wrong so nobody restores it as a "smaller step is
    more accurate" tidy-up. Two assertions: the step is below the representation
    floor (the mechanism), and the slope it yields is therefore worthless (the
    consequence). Neither depends on which way the rounding falls, so both hold
    on ARM (difference exactly 0.0) and on x86 (difference ±1 ULP).
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
    assert expected_change < ulp, (
        f"gamma={gamma}: the true change over h={h:.0e} is {expected_change:.3e}, no longer "
        f"below one ULP of the value ({ulp:.3e}). The step is now representable, so this "
        "is not a cancellation-floor demonstration any more — re-derive it"
    )

    difference = float(_total(gamma + h) - base)
    assert abs(difference) <= 2 * ulp, (
        f"gamma={gamma}: the h={h:.0e} difference is {difference:.3e}, more than 2 ULP "
        f"({2 * ulp:.3e}) — the subtraction is resolving real slope where it used to "
        "resolve only rounding"
    )

    rel = abs(difference / h - reference) / abs(reference)
    assert rel > 0.5, (
        f"gamma={gamma}: the h={h:.0e} one-sided FD is now only {rel:.1%} from the "
        f"converged derivative {reference:.4e}. It used to be useless there, which is the "
        "whole reason the shipped step is 1e-3 — re-measure the sweep in the docstring"
    )
