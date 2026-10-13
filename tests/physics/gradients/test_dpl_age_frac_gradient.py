# SPDX-License-Identifier: BSD-3-Clause
"""d(SED)/d(sfh_dpl_age_frac) is finite and nonzero near the age of the universe.

The fraction is a dimensionless onset in (0, 1] of age(z). At z = 2 the
gradient is taken at frac = 0.99, just inside the age of the universe, where
the onset is close to the Big Bang and the mass-conserving normalisation is
most sensitive to it (#2748).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.gradient

_Z = 2.0
_LOG_TOTAL_MASS = 10.0


def _model(ssp):
    return SEDModel.build(
        ssp_data=ssp,
        neb={"type": "none"},
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_frac": Uniform(0.5, 1.0, default=0.99),
            "log_total_mass": Fixed(_LOG_TOTAL_MASS),
        },
        redshift=Fixed(_Z),
    )


def test_sed_and_gradient_wrt_frac_are_finite_and_nonzero(synthetic_ssp_wide):
    """The SED sum and its frac-gradient are finite and nonzero at frac = 0.99, z = 2."""
    model = _model(synthetic_ssp_wide)

    def sed_sum(frac):
        return jnp.sum(model.predict_state({"sfh_dpl_age_frac": frac}).sed_intrinsic)

    value = sed_sum(0.99)
    grad = jax.grad(sed_sum)(0.99)

    assert bool(jnp.isfinite(value)) and float(value) > 0.0
    assert bool(jnp.isfinite(grad)), "d(SED)/d(frac) is not finite"
    assert float(grad) != 0.0, "d(SED)/d(frac) is exactly zero at frac=0.99, z=2"
