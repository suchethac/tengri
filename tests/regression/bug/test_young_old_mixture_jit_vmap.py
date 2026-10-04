# SPDX-License-Identifier: BSD-3-Clause
"""jit(vmap(predict_photometry)) equals the eager vmap with the young/old mixture on.

The age-interval mixture is a contraction over a 2-3 long interval axis. Written
as an einsum it miscompiled on XLA CPU when batched by ``vmap`` inside a larger
jitted graph (every band came back scaled by ~1e-4 for a zero-optical-depth
screen), which is the path ``Catalog.simulate`` takes. The mixture is now an
elementwise sum; this test pins the compiled batch against the eager batch.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug


def test_compiled_batch_matches_eager_batch(synthetic_ssp_wide, synthetic_tophat_obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={
                "type": "delayed",
                "tau_gyr": Uniform(0.5, 3.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
                "tau_bc": 0.0,
                "tau_diff": 0.0,
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
    params = {"sfh_delayed_tau_gyr": jnp.asarray([1.0, 2.0])}
    eager = np.asarray(jax.vmap(model.predict_photometry)(params))
    compiled = np.asarray(jax.jit(jax.vmap(model.predict_photometry))(params))
    np.testing.assert_allclose(compiled, eager, rtol=1e-10, atol=0.0)
