# SPDX-License-Identifier: BSD-3-Clause
"""At z = 0 the fractional dpl onset with frac = 1 reproduces the absolute Gyr onset.

``sfh_dpl_age_frac = 1`` maps to ``1 * age(z=0)``, the unrounded age of the
universe (#2748). The reference is the same build with the absolute Gyr name
fixed at that age, so the comparison is within one build and needs no
main-branch baseline. The registry default ``_AGE_UNIV_GYR`` is rounded to
three decimals and is not compared here.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.contract

_LOG_TOTAL_MASS = 10.0


def _sed_at_z0(ssp, sfh):
    model = SEDModel.build(
        ssp_data=ssp,
        neb={"type": "none"},
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Fixed(_LOG_TOTAL_MASS),
            **sfh,
        },
        redshift=Fixed(0.0),
    )
    return model.predict_state({}).sed_intrinsic


def test_frac_one_matches_gyr_at_age_of_universe(synthetic_ssp_wide):
    """frac = 1 and the Gyr onset set to age(z=0) give the same SED to 1e-10 relative."""
    age0 = float(age_at_z(0.0))
    via_frac = _sed_at_z0(synthetic_ssp_wide, {"age_frac": Fixed(1.0)})
    via_gyr = _sed_at_z0(synthetic_ssp_wide, {"age_gyr": Fixed(age0)})
    rel = jnp.max(jnp.abs(via_frac - via_gyr)) / jnp.max(jnp.abs(via_gyr))
    assert float(rel) <= 1e-10, f"relative SED difference {float(rel):.3e} > 1e-10"
