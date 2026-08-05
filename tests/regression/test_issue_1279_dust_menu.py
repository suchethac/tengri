# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1279: advertised dust emission types must build.

#1279 reported `dh02_ce01` (Dale & Helou 2002 + Chary & Elbaz 2001 cold dust)
as advertised in the build grammar but unbuildable. Measurement shows it builds
successfully via the lazy `DUST_EMISSION_MODELS` path even though absent from
`_REGISTRY`, so the reported issue does not reproduce. This test is the standing
guard that ensures every advertised dust-emission type is actually buildable,
preventing similar wiring gaps from being introduced.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_every_advertised_dust_emission_type_builds(synthetic_ssp, simple_observation):
    """Every advertised dust-emission type must be buildable.

    The menu returned by _valid_dust_emission_types() should be a subset of
    what the grammar builder can actually construct. This guards against
    advertising names that fail at build time.
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
