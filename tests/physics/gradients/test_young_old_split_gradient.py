# SPDX-License-Identifier: BSD-3-Clause
"""Gradients of photometry through the exact young/old split match central finite differences.

The young fraction of a node depends on the SFH through the age kernel (the
CIC scatter of ``contrib * S_bar``), so the gradient with respect to the SFH
parameters flows through it. A young SFH (age 12 Myr, so the cell holding the
10 Myr boundary carries most of the mass) is the case where an indicator
evaluated at node ages would have a zero or discontinuous gradient.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.gradient


def _model(ssp, obs, width):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Uniform(0.005, 0.05),
            "age_gyr": Uniform(0.006, 0.02),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.3),
            "transition_width_dex": width,
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


@pytest.mark.parametrize("width", [0.0, 0.3])
def test_photometry_gradient_matches_central_difference(
    synthetic_ssp_wide, synthetic_tophat_obs, width
):
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs, width)
    base = {
        "sfh_delayed_tau_gyr": jnp.asarray(0.02),
        "sfh_delayed_age_gyr": jnp.asarray(0.012),
    }

    def objective(p):
        return jnp.sum(jnp.log10(model.predict_photometry(p)))

    grad = jax.grad(objective)(base)
    for name, step in (("sfh_delayed_tau_gyr", 2e-4), ("sfh_delayed_age_gyr", 2e-5)):
        hi = dict(base, **{name: base[name] + step})
        lo = dict(base, **{name: base[name] - step})
        fd = (float(objective(hi)) - float(objective(lo))) / (2.0 * step)
        g = float(grad[name])
        assert np.isfinite(g), f"{name}: non-finite gradient through the young fraction"
        assert g != 0.0, f"{name}: identically zero gradient (split blind to the SFH)"
        np.testing.assert_allclose(g, fd, rtol=2e-3, atol=1e-8, err_msg=name)
