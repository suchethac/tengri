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

At ``h=1e-7`` the two evaluations are bit-identical and the difference is
*exactly* zero. The old step's error was not a fixed bias but ran from -10% to
+54% depending on where ``gamma`` sat, which is why checking a single step never
caught it.

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
    more accurate" tidy-up. At 1e-7 the two evaluations are bit-identical.
    """
    base = _total(gamma)
    assert float(_total(gamma + 1e-7) - base) == 0.0, (
        f"gamma={gamma}: a 1e-7 step no longer collapses to an exactly zero "
        "difference — the cancellation floor has moved, so re-derive the step"
    )
