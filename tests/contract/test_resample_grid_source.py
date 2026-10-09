# SPDX-License-Identifier: BSD-3-Clause
"""The resample decision reads the same rest grid whether or not it is traced.

``_predict_spectrum_on_grid`` resolves the flux-conserving flag once, before
tracing, from ``SEDModel.wavelengths``. The forward state publishes its own
``state.wave``. The two must be the same array for every component stack that
can change the forward grid (dust IR emission, radio, X-ray and AGN), or the
flag decided outside the trace would not describe the grid it is applied to.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Spectroscopy, builders

pytestmark = pytest.mark.contract


def _base_kwargs():
    return dict(
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


COMPONENT_STACKS = {
    "plain": {},
    "dust_ir": {"dust_emission": {"type": "dale2014"}},
    "radio": {"radio": {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}},
    "xray_agn": {
        "agn": {"type": "composable", "disc": builders.agn.disc.multicolor()},
        "xray": {"type": "simple"},
    },
}


@pytest.mark.parametrize("stack", sorted(COMPONENT_STACKS))
def test_state_wave_is_the_model_rest_grid(synthetic_ssp_wide, stack):
    """``state.wave`` is bit-identical to ``model.wavelengths`` for each stack."""
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=np.linspace(4000.0, 8500.0, 300)))
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        **COMPONENT_STACKS[stack],
        **_base_kwargs(),
    )
    params = model.spec.sample(jax.random.PRNGKey(1))
    state = model.predict_state(params)
    np.testing.assert_array_equal(np.asarray(state.wave), np.asarray(model.wavelengths))


@pytest.mark.parametrize("stack", sorted(COMPONENT_STACKS))
def test_resample_decision_agrees_on_both_grids(synthetic_ssp_wide, stack):
    """The conserving flag is the same whichever of the two grids is passed."""
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=np.linspace(4000.0, 8500.0, 300)))
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        **COMPONENT_STACKS[stack],
        **_base_kwargs(),
    )
    params = model.spec.sample(jax.random.PRNGKey(1))
    state = model.predict_state(params)
    z_ref = model._resample_z_ref()
    spectroscopy = model.observation.spectroscopy
    assert spectroscopy.resolve_conserving(state.wave, z_ref) == spectroscopy.resolve_conserving(
        model.wavelengths, z_ref
    )
