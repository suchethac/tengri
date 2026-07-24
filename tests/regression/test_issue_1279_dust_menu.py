# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1279: advertised dust emission types must build.

#1279: the build grammar advertised a dust-emission type `dh02_ce01` that
was not discoverable in the registry, causing silent failures when users
tried to build models with that type.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_every_advertised_dust_emission_type_builds(synthetic_ssp, simple_observation):
    """#1279: the build grammar's advertised dust-emission menu must be
    a subset of what the registry can actually construct.

    Every dust-emission type returned by the valid_dust_emission_types()
    menu must be successfully buildable via SEDModel.build().
    """
    from tengri import FIXED, SEDModel
    from tengri.parameters.groups import _valid_dust_emission_types

    # Get all advertised dust emission types
    for name in _valid_dust_emission_types():
        # Each should be buildable
        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=simple_observation,
            sfh={"type": "dpl"},
            dust={
                "type": "two_component",
                "all_params": FIXED,
                "emission": {"type": name, "all_params": FIXED},
            },
        )

        # Actually call predict to trigger component chain building
        params = {k: jnp.array(0.0) for k in model.spec.free_params}
        model.predict(params)
