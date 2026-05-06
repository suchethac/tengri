# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the BakedIn :class:`NebularSEDComponent` adapter.

Validates the no-op marker pattern: an adapter that declares zero free
parameters and does not transform the SED, but publishes a
``state.derived`` flag for downstream awareness.

Sister to :mod:`tests.integration.test_radio_igm_pipeline` etc. — kept
focused (no real nebular physics) so it runs in <1 s.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.nebular.component import (
    NebularSEDComponent,
    NebularSEDComponentConfig,
)
from tengri.core import PipelineState
from tengri.forward.orchestrator import merge_declared_parameters, run_components


@pytest.mark.unit
def test_nebular_baked_in_declares_zero_parameters():
    """The BakedIn backend has no free parameters by design."""
    nebular = NebularSEDComponent()
    assert nebular.declared_parameters() == []


@pytest.mark.unit
def test_nebular_does_not_modify_sed_intrinsic():
    """BakedIn nebular is a no-op on the SED — emission is in the SSP grid."""
    wave = jnp.linspace(1000.0, 10000.0, 64)
    intrinsic = jnp.ones_like(wave) * 1e30
    state = PipelineState(wave=wave, sed_intrinsic=intrinsic)

    out = NebularSEDComponent().apply(state, {"redshift": 0.0})

    assert out.sed_intrinsic is state.sed_intrinsic, (
        "BakedIn must not allocate a new sed_intrinsic; it should be untouched"
    )
    assert out.sed_attenuated is None  # nothing attenuated yet
    assert out.sed_observed is None


@pytest.mark.unit
def test_nebular_publishes_sed_keys_for_baked_in():
    """``sed_nebular`` and ``sed_shock`` are always published (zeros for BakedIn)."""
    wave = jnp.linspace(1000.0, 10000.0, 32)
    state = PipelineState(wave=wave)
    out = NebularSEDComponent().apply(state, {"redshift": 0.0})
    assert "sed_nebular" in out.derived
    assert "sed_shock" in out.derived
    assert jnp.all(out.derived["sed_nebular"] == 0.0)
    assert jnp.all(out.derived["sed_shock"] == 0.0)


@pytest.mark.unit
def test_nebular_in_orchestrator_chain():
    """Nebular plugs into the orchestrator with no special handling."""
    from tengri.components.dust.component import DustAttenuationSEDComponent
    from tengri.components.radio.component import RadioSEDComponent

    wave = jnp.linspace(1000.0, 100000.0, 128)
    state = PipelineState(
        wave=wave,
        sed_intrinsic=jnp.ones_like(wave) * 1e30,
        derived={"L_ir": 1e44, "L_agn_bol": 0.0, "log_mstar": 10.0},
    )
    params = {
        "redshift": 0.0,
        "radio_q_ir": 2.64,
        "radio_alpha_sf": 0.8,
        "radio_loudness": 0.0,
        "radio_alpha_agn": 0.7,
        "radio_T_e": 1e4,
        "radio_alpha_ff": -0.1,
        "dust_tau_v": 0.3,
    }

    final = run_components(
        [NebularSEDComponent(), RadioSEDComponent(), DustAttenuationSEDComponent()],
        state,
        params,
    )

    # All three components must have run successfully.
    assert "sed_nebular" in final.derived
    assert "sed_shock" in final.derived
    assert "sed_radio" in final.derived
    assert final.sed_attenuated is not None


@pytest.mark.unit
def test_merge_with_zero_param_component():
    """``merge_declared_parameters`` handles a zero-parameter component cleanly."""
    nebular = NebularSEDComponent()
    merged = merge_declared_parameters([nebular])
    assert merged == {}, "BakedIn nebular contributes no priors"


@pytest.mark.unit
def test_unknown_backend_string_raises():
    """An unknown backend identifier surfaces a clear error.

    Cue / CloudyGrid / shock are now supported (Phase II-3 closure);
    only truly unknown identifiers should fail.
    """
    bad = NebularSEDComponent(
        config=NebularSEDComponentConfig(name="nebular", backend="not_a_backend")
    )
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        bad.declared_parameters()
