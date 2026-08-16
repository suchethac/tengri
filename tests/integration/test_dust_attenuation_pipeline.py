# SPDX-License-Identifier: BSD-3-Clause
"""Numerical-equivalence tests for the dust-attenuation Phase II-1 adapter.

Asserts that running ``[DustAttenuationSEDComponent]`` through the
orchestrator multiplies ``sed_intrinsic`` by ``exp(-tau_v * k(λ))``
exactly. The first adapter that *transforms* the SED rather than adding
to it: writes ``sed_attenuated`` from ``sed_intrinsic``.

Mirrors :mod:`tests.integration.test_xray_pipeline` for layout. Also
exercises a **four-adapter chain** (Radio + Dust + X-ray + IGM) that
demonstrates the orchestrator can compose additive emitters with a
transforming attenuator without any contract violations.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from tengri.components.dust.attenuation import calzetti
from tengri.components.dust.component import (
    DustAttenuationSEDComponent,
    DustAttenuationSEDComponentConfig,
)
from tengri.forward.orchestrator import default_params_dict, run_components
from tengri.protocols import ForwardState

REL_TOL = 1e-10


@pytest.mark.parametrize("tau_v", [0.0, 0.1, 0.5, 1.5, 3.0])
def test_orchestrator_matches_direct_attenuation(tau_v):
    """Pipeline output equals sed_intrinsic * exp(-tau_v * k_calzetti(λ))."""
    wave = jnp.linspace(1000.0, 22000.0, 256)  # Calzetti valid range 0.12-2.2 μm
    intrinsic = jnp.ones_like(wave) * 1e30  # placeholder L_nu

    initial_state = ForwardState(
        wave=wave,
        sed_intrinsic=intrinsic,
    )
    params = {"redshift": 0.0, "dust_tau_v": tau_v}

    final = run_components([DustAttenuationSEDComponent()], initial_state, params)

    expected_k = calzetti(wave)
    expected = intrinsic * jnp.exp(-tau_v * expected_k)

    assert final.sed_attenuated is not None
    assert jnp.allclose(final.sed_attenuated, expected, rtol=REL_TOL, atol=0.0)
    # The attenuation factor is published for downstream readers.
    assert "dust_attenuation_factor" in final.derived
    assert jnp.allclose(
        final.derived["dust_attenuation_factor"],
        jnp.exp(-tau_v * expected_k),
        rtol=REL_TOL,
        atol=0.0,
    )


def test_dust_is_noop_when_sed_intrinsic_is_none():
    """No upstream emitter → dust has nothing to attenuate."""
    wave = jnp.linspace(1000.0, 22000.0, 64)
    state = ForwardState(wave=wave)
    out = DustAttenuationSEDComponent().apply(state, {"dust_tau_v": 0.5, "redshift": 0.0})
    assert out.sed_attenuated is None
    assert out.sed_intrinsic is None


def test_tau_zero_is_identity():
    """tau_v = 0 must leave sed_intrinsic unchanged in sed_attenuated."""
    wave = jnp.linspace(1000.0, 22000.0, 64)
    intrinsic = jnp.ones_like(wave) * 5.0
    state = ForwardState(wave=wave, sed_intrinsic=intrinsic)
    out = DustAttenuationSEDComponent().apply(state, {"dust_tau_v": 0.0, "redshift": 0.0})
    assert jnp.allclose(out.sed_attenuated, intrinsic, rtol=REL_TOL, atol=0.0)


def test_smc_law_via_config():
    """Choosing a non-default law via config still works."""
    wave = jnp.linspace(1000.0, 22000.0, 64)
    intrinsic = jnp.ones_like(wave) * 1e30
    state = ForwardState(wave=wave, sed_intrinsic=intrinsic)

    smc_dust = DustAttenuationSEDComponent(
        config=DustAttenuationSEDComponentConfig(name="dust", law="smc")
    )

    # Just verify it runs and produces finite, attenuating output.
    out = smc_dust.apply(state, {"dust_tau_v": 0.5, "redshift": 0.0})
    assert out.sed_attenuated is not None
    chex.assert_tree_all_finite(out.sed_attenuated)
    # SMC steeper than Calzetti at UV — UV attenuation must exceed
    # optical attenuation.
    uv_a = float(jnp.mean(out.sed_attenuated[:8])) / float(jnp.mean(intrinsic[:8]))
    opt_a = float(jnp.mean(out.sed_attenuated[-8:])) / float(jnp.mean(intrinsic[-8:]))
    uv_attenuation = uv_a
    optical_attenuation = opt_a
    assert uv_attenuation < optical_attenuation, (
        "SMC law should attenuate UV more aggressively than optical"
    )


def test_four_adapter_chain_runs_end_to_end():
    """Radio + Dust + X-ray + IGM composed in a single pipeline.

    Asserts the four-component chain produces a finite final state with
    every published-derived key present and IGM-modulated sed_observed
    visibly attenuated at z=8 (above igm_z_mid=7).
    """
    from tengri.components.igm.component import IGMSEDComponent
    from tengri.components.radio.component import RadioSEDComponent
    from tengri.components.xray.component import XRaySEDComponent

    # Log-spaced grid so the UV/Lyα region has points (where IGM bites).
    wave = jnp.logspace(0, 9, 256)
    state = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.ones_like(wave) * 1e30,
        sed_observed=jnp.ones_like(wave) * 1e30,
        derived={
            "L_ir": 1e44,
            "L_agn_bol": 1e44,
            "log_mstar": 10.5,  # XRay reads this, exponentiates internally
            "sfr": 5.0,
        },
    )

    chain = [
        RadioSEDComponent(),
        DustAttenuationSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]

    # Declared defaults, with the departures this test chose spelled out. The
    # literal this replaces named nineteen keys, matched the declarations on
    # every one of them, and still went stale when xray_det_hmxb gained a
    # reader (#1832).
    params = default_params_dict(
        chain,
        overrides={
            "redshift": 8.0,
            "dust_tau_v": 0.5,
            "xray_delta_alpha_ox": -1.4,
            "xray_log_nh": 21.0,  # AGN corona photoelectric absorption
        },
    )

    final = run_components(
        chain,
        state,
        params,
    )

    assert final.sed_intrinsic is not None
    assert final.sed_attenuated is not None
    assert final.sed_observed is not None
    chex.assert_tree_all_finite(final.sed_intrinsic)
    chex.assert_tree_all_finite(final.sed_attenuated)
    chex.assert_tree_all_finite(final.sed_observed)
    # All four components published their derived quantities.
    for key in ("sed_radio", "sed_xray", "dust_attenuation_factor"):
        assert key in final.derived, f"{key} missing from final.derived"
    # Dust must have actually attenuated (not just copied).
    assert jnp.any(final.sed_attenuated < final.sed_intrinsic)
    # IGM at z=8 must have applied non-trivial transmission.
    assert jnp.any(final.sed_observed < state.sed_observed)
