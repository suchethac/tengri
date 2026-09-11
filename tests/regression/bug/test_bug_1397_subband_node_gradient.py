# SPDX-License-Identifier: BSD-3-Clause
"""Regression: sub-band node wavelengths must stay differentiable (#1397).

The WavePrecomp sub-band node is a flux-weighted mean,
``lambda_k = sub_num / sub_phi``, guarded by the standard double-where so a
zero-weight sub-band falls back to the band effective wavelength::

    live = sub_phi != 0.0
    jnp.where(live, sub_num / jnp.where(live, sub_phi, 1.0), eff_waves)

That protects the **value** but not the **derivative**. The quotient rule needs
``sub_phi**2``, and once ``sub_phi`` falls below ``sqrt(2.2e-308) ~ 1.5e-154``
that square underflows to exactly zero -- so the gradient divides by zero while
the value is still a perfectly finite ratio of two tiny numbers. Testing
``!= 0.0`` cannot catch it: every one of those weights is nonzero.

A narrow SFH is enough to get there. Measured on
``recipes.mock_recovery_minimal()``, the smallest nonzero sub-band weight
collapses with the tsnorm width::

    width 2.0 Gyr -> 1.4e+19      width 0.30 Gyr -> 5.0e-170
    width 1.0 Gyr -> 2.6e+05      width 0.25 Gyr -> 2.5e-254
    width 0.5 Gyr -> 2.2e-47      width 0.20 Gyr -> 1.2e-257

``width`` is a free parameter with prior ``Uniform(0.2, 5.0)``, so the model's
own prior box contained a region where every gradient-based method -- MAP, HMC,
NUTS -- got NaN. That is what made the MAP diverge in #1397, which produced the
NaN metric that the preconditioner then misreported as "not positive definite".

The fix floors the liveness test instead of testing against exact zero. It is a
numerical no-op: photometry over 40 prior draws is bit-identical either way,
because a sub-band carrying 1e-250 of the flux contributes nothing measurable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    Observation,
    Photometry,
    SEDModel,
    WavePrecomp,
    recipes,
)

pytestmark = [pytest.mark.regression_bug, pytest.mark.gradient]

_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]
# Inside the tsnorm prior Uniform(0.2, 5.0). The lower half is where sub-band
# weights underflow past the square-representable limit.
_WIDTHS_GYR = (2.0, 1.0, 0.5, 0.3, 0.25, 0.2)


@pytest.fixture(scope="module")
def lut_model(ssp_data_fsps):
    """The #1397 model: mock_recovery_minimal, which ships WavePrecomp."""
    obs = Observation(photometry=Photometry.from_names(_BANDS))
    return SEDModel.build(
        ssp_data=ssp_data_fsps, observation=obs, **recipes.mock_recovery_minimal()
    )


def _params(model, width_gyr):
    return {
        **model.spec.get_fixed_values(),
        "sfh_tsnorm_log_total_mass": 10.0,
        "sfh_tsnorm_peak_lbt_gyr": 5.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 1.0,
        "met_logzsol": 0.0,
        "dust_tau_bc": 0.1,
        "sfh_tsnorm_width_gyr": width_gyr,
    }


@pytest.mark.parametrize("width_gyr", _WIDTHS_GYR)
def test_lut_photometry_gradient_is_finite_across_the_width_prior(lut_model, width_gyr):
    """LOAD-BEARING. Neuter: restore ``live = sub_phi != 0.0``.

    Without the floor this is NaN for width <= 0.3 Gyr, well inside the prior.
    """

    def flux_sum(w):
        return jnp.sum(lut_model.predict_photometry(_params(lut_model, w)))

    grad = float(jax.grad(flux_sum)(width_gyr))
    assert np.isfinite(grad), (
        f"d(photometry)/d(width) is {grad} at width={width_gyr} Gyr, inside the "
        "tsnorm prior Uniform(0.2, 5.0)"
    )
    assert np.any(grad != 0.0), (
        "`grad` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def test_the_value_was_never_the_problem(lut_model):
    """Pins the premise: the forward value is finite even where the gradient was not.

    Guards against a future 'fix' that merely clamps the output — the defect was
    always in the derivative.
    """
    for width_gyr in _WIDTHS_GYR:
        flux = np.asarray(lut_model.predict_photometry(_params(lut_model, width_gyr)))
        assert np.all(np.isfinite(flux)), f"value non-finite at width={width_gyr}"


def test_the_floor_does_not_change_the_photometry(ssp_data_fsps):
    """The floor must be a numerical no-op where the weights are usable.

    Compares the LUT path against the exact wave-grid path, which has no
    sub-band decomposition at all and so is unaffected by the floor. They agree
    to the LUT's own accuracy, which they would not if the floor were discarding
    sub-bands that carry real flux.
    """
    obs = Observation(photometry=Photometry.from_names(_BANDS))
    recipe = recipes.mock_recovery_minimal()
    without_approx = {k: v for k, v in recipe.items() if k != "approx"}
    lut = SEDModel.build(
        ssp_data=ssp_data_fsps, observation=obs, approx=WavePrecomp(), **without_approx
    )
    exact = SEDModel.build(ssp_data=ssp_data_fsps, observation=obs, approx=None, **without_approx)
    for width_gyr in (2.0, 1.0, 0.5):
        a = np.asarray(lut.predict_photometry(_params(lut, width_gyr)))
        b = np.asarray(exact.predict_photometry(_params(exact, width_gyr)))
        np.testing.assert_allclose(
            a, b, rtol=0.05, err_msg=f"LUT and exact paths disagree at width={width_gyr}"
        )
