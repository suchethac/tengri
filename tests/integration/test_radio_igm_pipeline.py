# SPDX-License-Identifier: BSD-3-Clause
"""Numerical-equivalence tests for the radio + IGM Phase II-1 adapters.

Asserts that running ``[RadioSEDComponent, IGMSEDComponent]`` through
:func:`tengri.forward.orchestrator.run_components` produces the same
SED as calling :func:`radio_total` and :func:`igm_transmission`
directly. Holds the orchestrator + adapters honest as a pair: any
future Protocol revision that breaks this contract is caught here, not
in a science fit weeks later.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.igm.component import IGMSEDComponent
from tengri.components.igm.igm import igm_transmission
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.radio.radio import radio_total
from tengri.protocols import PipelineState
from tengri.forward.orchestrator import run_components

REL_TOL = 1e-10


@pytest.mark.parametrize(
    ("z", "L_ir", "L_agn_bol", "log_mstar"),
    [
        (0.1, 1e44, 0.0, 10.0),
        (0.5, 1e45, 1e44, 10.5),
        (2.0, 1e46, 5e45, 11.0),
        (4.0, 1e44, 0.0, 9.5),
        (6.0, 1e43, 1e44, 9.0),
    ],
)
def test_orchestrator_matches_direct_calls(z, L_ir, L_agn_bol, log_mstar):
    """Pipeline output equals direct radio_total + igm_transmission."""
    wave = jnp.linspace(1e3, 1e9, 1024)
    seed_observed = jnp.ones_like(wave) * 1e30  # placeholder F_nu

    initial_state = PipelineState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        sed_observed=seed_observed,
        derived={"L_ir": L_ir, "L_agn_bol": L_agn_bol, "log_mstar": log_mstar},
    )

    params = {
        "redshift": z,
        "radio_q_ir": 2.64,
        "radio_alpha_sf": 0.8,
        "radio_loudness": 0.0,
        "radio_alpha_agn": 0.7,
        "radio_T_e": 1e4,
        "radio_alpha_ff": -0.1,
        "igm_z_mid": 7.0,
        "igm_dz": 0.5,
        "igm_log_nhi": 20.0,
    }

    final = run_components(
        [RadioSEDComponent(), IGMSEDComponent()],
        initial_state,
        params,
    )

    expected_radio = radio_total(
        wave,
        L_ir=L_ir,
        L_agn_bol=L_agn_bol,
        q_ir=2.64,
        alpha_sf=0.8,
        radio_loudness=0.0,
        alpha_agn=0.7,
        sfr_mode="bell2003",
        log_mstar=log_mstar,
        redshift=z,
        include_freefree=True,
        T_e=1e4,
        alpha_ff=-0.1,
    )
    expected_T = igm_transmission(wave * (1.0 + z), z)
    expected_observed = seed_observed * expected_T

    assert jnp.allclose(final.sed_intrinsic, expected_radio, rtol=REL_TOL, atol=0.0)
    assert jnp.allclose(final.sed_observed, expected_observed, rtol=REL_TOL, atol=0.0)


def test_pipeline_preserves_input_state_immutability():
    """The orchestrator must not mutate the input PipelineState."""
    wave = jnp.linspace(1e3, 1e9, 64)
    initial = PipelineState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        sed_observed=jnp.ones_like(wave),
        derived={"L_ir": 1e44, "L_agn_bol": 0.0, "log_mstar": 10.0},
    )
    snapshot_intrinsic = initial.sed_intrinsic
    snapshot_observed = initial.sed_observed

    _ = run_components(
        [RadioSEDComponent(), IGMSEDComponent()],
        initial,
        {
            "redshift": 1.0,
            "radio_q_ir": 2.64,
            "radio_alpha_sf": 0.8,
            "radio_loudness": 0.0,
            "radio_alpha_agn": 0.7,
            "radio_T_e": 1e4,
            "radio_alpha_ff": -0.1,
            "igm_z_mid": 7.0,
            "igm_dz": 0.5,
            "igm_log_nhi": 20.0,
        },
    )

    assert jnp.array_equal(initial.sed_intrinsic, snapshot_intrinsic)
    assert jnp.array_equal(initial.sed_observed, snapshot_observed)


def test_radio_no_dust_upstream_falls_back_to_zero():
    """When state.derived has no L_ir, radio uses its documented fallback (0)."""
    wave = jnp.linspace(1e3, 1e9, 64)
    state = PipelineState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        sed_observed=jnp.ones_like(wave),
    )
    radio = RadioSEDComponent()
    params = {
        "redshift": 0.0,
        "radio_q_ir": 2.64,
        "radio_alpha_sf": 0.8,
        "radio_loudness": 0.0,
        "radio_alpha_agn": 0.7,
        "radio_T_e": 1e4,
        "radio_alpha_ff": -0.1,
    }
    out = radio.apply(state, params)

    expected = radio_total(
        wave,
        L_ir=0.0,
        L_agn_bol=0.0,
        q_ir=2.64,
        alpha_sf=0.8,
        radio_loudness=0.0,
        alpha_agn=0.7,
        sfr_mode="bell2003",
        log_mstar=10.0,
        redshift=0.0,
        include_freefree=True,
        T_e=1e4,
        alpha_ff=-0.1,
    )
    assert jnp.allclose(out.sed_intrinsic, expected, rtol=REL_TOL, atol=0.0)


def test_igm_noop_when_sed_observed_is_none():
    """IGM is a no-op if no upstream produced sed_observed."""
    wave = jnp.linspace(1e3, 1e9, 64)
    state = PipelineState(wave=wave)
    igm = IGMSEDComponent()
    out = igm.apply(state, {"redshift": 1.0, "igm_z_mid": 7.0, "igm_dz": 0.5, "igm_log_nhi": 20.0})
    assert out.sed_observed is None
