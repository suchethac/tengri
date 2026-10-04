# SPDX-License-Identifier: BSD-3-Clause
r"""The ``nthcomp`` gamma tangent must match a converged central difference.

``_nthcomp_interp_jvp`` returns the exact piecewise-linear slope of the template
interpolant. A converged central difference of the forward value (step 1e-3, on
the plateau where truncation and rounding are both far below 1%) is the independent
reference these tests hold it to.

The forward value is computed with the interpolation coordinates in the caller's
dtype, so a step as small as 1e-7 in ``gamma`` is resolved as real slope. When the
coordinates were rounded to float32 the same step moved the value by less than half
a ULP and the difference came back as rounding (exactly 0.0 on macOS/ARM, +-1 ULP on
x86), which is what the last test below pins.

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
def test_a_tiny_gamma_step_is_resolved_as_slope(gamma):
    """A 1e-7 step in ``gamma`` changes the value by its slope, not by rounding.

    Holds only while the interpolation coordinate is not quantized: at float32
    precision the true change over this step is below half a ULP of the value and
    the one-sided difference is 0 or +-1 ULP, a 100-300% wrong slope.
    """
    h = 1e-7
    reference = _central(gamma)
    assert np.isfinite(reference) and reference != 0.0, f"setup: bad reference at gamma={gamma}"

    difference = float(_total(gamma + h) - _total(gamma))
    rel = abs(difference / h - reference) / abs(reference)
    assert rel < 1e-2, (
        f"gamma={gamma}: the h={h:.0e} one-sided FD is {rel:.1%} from the converged "
        f"derivative {reference:.4e}; the coordinate is being quantized again"
    )
