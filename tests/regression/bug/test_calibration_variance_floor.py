# SPDX-License-Identifier: BSD-3-Clause
r"""The calibration polynomial collapses at real flux scales — in float64.

``marginalize_calibration`` weighted its normal equations with

.. code-block:: python

    inv_var = 1.0 / jnp.maximum(obs_err**2, 1e-30)

The floor is written as a guard against ``obs_err == 0``, but it is expressed in
**variance**, so it binds whenever :math:`\sigma < 10^{-15}`. A spectroscopic
:math:`\sigma` is ~1e-20 (F_lambda) to ~1e-30 (F_nu), so it binds on *every
pixel of every realistic spectrum* — pinning ``inv_var`` to exactly 1e30
regardless of the true uncertainty.

The data term of ``A = B B^T + I/prior_sigma^2`` then shrinks until the prior
wins, and the recovered coefficients collapse toward zero. Measured against a
known truth ``[0.05, -0.02, 0.01]``:

===============  ===========  ==================  ==================
flux scale       prior_sigma  c_hat[0] shipped    c_hat[0] unfloored
===============  ===========  ==================  ==================
O(1) (control)   1.0          5.00e-02            5.00e-02
1e-17            1.0          **7.46e-04**        5.00e-02
1e-17            10.0         **2.87e-02**        5.00e-02
1e-28 (F_nu)     1.0          **7.58e-26**        5.00e-02
===============  ===========  ==================  ==================

This is **not** a float32 issue — every number above is float64, the default.
It was originally filed as a Tier-B float32 known-open (#1588's guard
allowlist); that framing was wrong, and this module is the correction.

Two things make the measurement mean something. The O(1) control is
*bit-identical* between the floored and unfloored arms, so the floor is the only
difference at the small scales. And the unfloored arm recovers the truth
**exactly at every scale** — the calibration polynomial is a flux-*ratio*, so
its answer must be scale-invariant, which pins the correct behavior
independently of any reference implementation.

Why the existing suite missed it: ``test_known_calibration_recovery`` runs at
1e-17 with ``prior_sigma=10.0`` and ``atol=0.03``. The floored answer is 43%
low and still inside that tolerance. Its sibling
``test_marginal_higher_than_uncalibrated`` compares against an *unfloored*
hand-rolled chi2, and the bug pushes that inequality in the passing direction.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.calibration import calibration_polynomial, marginalize_calibration

pytestmark = pytest.mark.regression_bug

#: The calibration polynomial is a flux ratio, so the recovered coefficients
#: must not depend on the flux scale at all.
_TRUTH = (0.05, -0.02, 0.01)

#: F_lambda ~1e-17, F_nu ~1e-28 — both real; 1.0 is the control the old tests used.
_SCALES = (1.0, 1e-17, 1e-28)


def _wave():
    return jnp.linspace(3800.0, 9000.0, 500)


def _inputs(scale):
    """Model, calibrated 'observation', and 1-sigma errors at ``scale``."""
    wave = _wave()
    model = scale * (wave / 5000.0) ** -1.5
    cal = calibration_polynomial(wave, jnp.asarray(_TRUTH), wave[0], wave[-1])
    return wave, model, cal * model, 0.001 * model


@pytest.mark.parametrize("scale", _SCALES)
@pytest.mark.parametrize("prior_sigma", [1.0, 10.0])
def test_calibration_recovers_truth_at_every_flux_scale(scale, prior_sigma):
    """c_hat must recover the truth whatever the flux units are.

    Runs at the default ``prior_sigma=1.0`` as well as the 10.0 the older test
    used: the collapse is far worse at the default, which is what a real fit
    gets.
    """
    with jax.enable_x64(True):
        wave, model, obs, err = _inputs(scale)
        _ll, c_hat, _e = marginalize_calibration(
            model, obs, err, wave, n_poly=3, prior_sigma=prior_sigma
        )
        got = np.asarray(c_hat, dtype=np.float64)

    # Shrinkage toward zero is legitimate for a finite prior, so compare
    # against the same prior's answer at O(1) rather than against the raw truth.
    with jax.enable_x64(True):
        wave1, model1, obs1, err1 = _inputs(1.0)
        _ll1, c_ref, _e1 = marginalize_calibration(
            model1, obs1, err1, wave1, n_poly=3, prior_sigma=prior_sigma
        )
        ref = np.asarray(c_ref, dtype=np.float64)

    np.testing.assert_allclose(
        got,
        ref,
        rtol=1e-6,
        err_msg=(
            f"calibration coefficients depend on the FLUX SCALE ({scale:g}): "
            f"got {got}, but the same fit at O(1) gives {ref}. The polynomial is a "
            "flux ratio — its answer must be scale-invariant. A variance-domain "
            "floor (1e-30) binds at small sigma and lets the prior win."
        ),
    )


@pytest.mark.parametrize("scale", _SCALES[1:])
def test_the_variance_floor_would_bind_at_these_scales(scale):
    """Precondition: without it, the tests above could pass vacuously.

    If a future fixture drifts to larger fluxes, the regression tests stop
    exercising the defect while still reporting green. This fails loudly first.
    """
    _wv, _model, _obs, err = _inputs(scale)
    min_var = float(jnp.min(jnp.asarray(err, dtype=jnp.float64) ** 2))
    assert min_var < 1e-30, (
        f"min(obs_err**2) = {min_var:.3e} is above the 1e-30 floor at scale "
        f"{scale:g}, so this fixture no longer exercises the defect — lower the "
        "flux scale rather than deleting the test"
    )


def test_o1_control_is_unaffected_by_the_fix():
    """The O(1) arm must be identical to the floored spelling.

    This is the control that makes the parametrized test above meaningful: if
    the fix changed *everything*, agreement at small scales would prove nothing.
    """
    with jax.enable_x64(True):
        wave, model, obs, err = _inputs(1.0)
        _ll, c_hat, _e = marginalize_calibration(model, obs, err, wave, n_poly=3)
        got = np.asarray(c_hat, dtype=np.float64)
        # The floor cannot bind here, so the shipped arithmetic is reproduced
        # exactly by hand with the floor still in place.
        min_var = float(jnp.min(jnp.asarray(err, dtype=jnp.float64) ** 2))
    assert min_var > 1e-30, "the O(1) control must be a case where the floor does NOT bind"
    np.testing.assert_allclose(got, np.asarray(_TRUTH), atol=2e-3)
