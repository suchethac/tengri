# SPDX-License-Identifier: BSD-3-Clause
"""Dust emission can use an explicit IR luminosity without attenuation."""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri.components.dust.emission._physics import integrate_lnu_over_nu
from tengri.config.exceptions import ParameterError
from tengri.forward.sed_model import WavePrecomp
from tengri.utils.sed_quantities import LOG10_L_SUN

pytestmark = pytest.mark.contract


def _build(ssp, observation, *, emission=None, approx=None):
    groups = {
        "sfh": {"all_params": tengri.Fixed(tengri.DEFAULT)},
        "dust_attenuation": {"type": "none"},
        "redshift": tengri.Fixed(0.0),
    }
    if emission is not None:
        groups["dust_emission"] = emission
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        approx=approx,
        **groups,
    )


def test_dust_off_without_emitter_still_builds_and_emitter_needs_luminosity(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """No emission group needs no IR parameter; a standalone emitter does."""
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    state = model.predict_state({})
    assert state.derived.L_ir is None

    with pytest.raises(ParameterError, match="explicit dust_log_L_ir"):
        _build(
            synthetic_ssp_wide,
            synthetic_tophat_obs,
            emission={"type": "haro11", "other_params": tengri.Fixed(tengri.DEFAULT)},
        )


@pytest.mark.parametrize("emission_type", ["haro11", "modified_blackbody"])
def test_emission_uses_explicit_ir_luminosity_without_absorbed_budget(
    synthetic_ssp_wide, synthetic_tophat_obs, emission_type
):
    """A standalone emitter integrates to the request without an absorbed budget."""
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        emission={
            "type": emission_type,
            "log_L_ir": tengri.Fixed(11.0),
            "other_params": tengri.Fixed(tengri.DEFAULT),
        },
    )
    state = model.predict_state({})
    wave = np.asarray(state.wave, dtype=np.float64)
    sed_ir = np.asarray(state.derived.sed_dust_ir, dtype=np.float64)
    log_l_ir = float(np.asarray(state.derived.log_L_ir))
    l_ir = float(np.asarray(state.derived.L_ir))

    assert log_l_ir == pytest.approx(11.0 + LOG10_L_SUN, abs=1e-8)
    balance_rtol = 1e-7 if emission_type == "haro11" else 5e-6
    assert float(integrate_lnu_over_nu(sed_ir, wave)) / l_ir == pytest.approx(
        1.0, rel=balance_rtol
    )
    assert state.derived.L_absorbed is None
    assert state.derived.log_L_absorbed is None
    if emission_type == "haro11":
        assert np.all(sed_ir[wave <= 50_000.0] == 0.0)


def test_diffuse_screen_requires_active_attenuation(synthetic_ssp_wide, synthetic_tophat_obs):
    """A diffuse emission screen cannot be requested when no dust screen is built."""
    with pytest.raises(ValueError, match="requires dust_attenuation to be active"):
        _build(
            synthetic_ssp_wide,
            synthetic_tophat_obs,
            emission={
                "type": "haro11",
                "log_L_ir": tengri.Fixed(11.0),
                "diffuse_screen": True,
                "other_params": tengri.Fixed(tengri.DEFAULT),
            },
        )


def test_standalone_budget_does_not_publish_projection_luts(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """A luminosity-only provider must not add photometry or spectrum LUT keys."""
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        approx=WavePrecomp(),
        emission={
            "type": "haro11",
            "log_L_ir": tengri.Fixed(11.0),
            "other_params": tengri.Fixed(tengri.DEFAULT),
        },
    )
    state = model.predict_state({})

    assert float(np.asarray(state.derived.log_L_ir)) == pytest.approx(11.0 + LOG10_L_SUN)
    assert not any(key.startswith("dust_ir_budget_") for key in state.derived._extras)
