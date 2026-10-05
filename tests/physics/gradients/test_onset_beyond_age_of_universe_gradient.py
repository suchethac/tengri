# SPDX-License-Identifier: BSD-3-Clause
"""SED and dSED/d(onset) stay finite and nonzero on both sides of age(z=6).

A z-capped onset/age/peak-time parameter's declared ceiling is only ever
guaranteed correct against the LOWEST redshift a free ``redshift`` prior
admits (:func:`~tengri.parameters.groups._narrow_free_priors_to_z`); at a
fixed high redshift, or at the upper end of a free one, a value inside that
declared range can still place star formation before the Big Bang
(:class:`~tengri.config.exceptions.FreeRedshiftOnsetCeilingWarning`, #2521).

This does not make the forward model unsafe to differentiate: the mass a
too-old onset would place before the Big Bang is truncated and reassigned
inside ``_mass_conserving_total`` (#2521), which keeps the total finite and
keeps the gradient wired through the surviving (post-Big-Bang) support rather
than an ``Inf``, a ``NaN``, or an exact zero. That is the property this test
pins directly at z=6 (age 0.9342 Gyr): one ``dpl`` onset draw inside that
age, one draw past it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.gradient

_Z = 6.0
_AGE_Z_GYR = float(age_at_z(_Z))  # 0.9342 Gyr
_LOG_TOTAL_MASS = 10.0


def _model(ssp):
    return SEDModel.build(
        ssp_data=ssp,
        neb={"type": "none"},
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_gyr": Uniform(0.05, 6.0),
            "log_total_mass": Fixed(_LOG_TOTAL_MASS),
        },
        redshift=Fixed(_Z),
    )


@pytest.mark.parametrize(
    "age_gyr,side",
    [
        pytest.param(0.5, "inside", id="inside-age-of-universe"),
        pytest.param(2.0, "outside", id="outside-age-of-universe"),
    ],
)
def test_sed_and_gradient_are_finite_and_nonzero(synthetic_ssp_wide, age_gyr, side):
    """dpl age_gyr on both sides of age_at_z(6) = 0.9342 Gyr."""
    if side == "inside":
        assert age_gyr < _AGE_Z_GYR
    else:
        assert age_gyr > _AGE_Z_GYR

    model = _model(synthetic_ssp_wide)

    def sed_sum(a):
        return jnp.sum(model.predict_state({"sfh_dpl_age_gyr": a}).sed_intrinsic)

    value = sed_sum(age_gyr)
    grad = jax.grad(sed_sum)(age_gyr)

    assert bool(jnp.isfinite(value)), f"SED sum is not finite at age_gyr={age_gyr} ({side})"
    assert value > 0.0, f"SED sum is not positive at age_gyr={age_gyr} ({side})"
    assert bool(jnp.isfinite(grad)), f"d(SED)/d(age_gyr) is not finite at age_gyr={age_gyr}"
    assert grad != 0.0, f"d(SED)/d(age_gyr) is exactly zero at age_gyr={age_gyr} ({side})"


@pytest.mark.parametrize(
    "age_gyr,side",
    [
        pytest.param(0.5, "inside", id="inside-age-of-universe"),
        pytest.param(2.0, "outside", id="outside-age-of-universe"),
    ],
)
def test_formed_mass_still_equals_the_request(synthetic_ssp_wide, age_gyr, side):
    """``_mass_conserving_total`` pins formed mass to the request either side."""
    model = _model(synthetic_ssp_wide)
    state = model.predict_state({"sfh_dpl_age_gyr": age_gyr})
    log_mass = float(state.derived["log_mstar_formed"])
    assert log_mass == pytest.approx(_LOG_TOTAL_MASS, abs=1e-6), (
        f"log_mstar_formed={log_mass} != requested {_LOG_TOTAL_MASS} at "
        f"age_gyr={age_gyr} ({side} age_at_z({_Z})={_AGE_Z_GYR:.4f})"
    )
