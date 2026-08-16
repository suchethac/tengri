# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end test for the public ``tengri.pipeline`` namespace.

This test exercises the full Phase II-1 surface as a user would import
it: one canonical import path, five adapters chained, parameters merged
through :func:`merge_declared_parameters`, the chain executed via
:func:`run_components`. It is intentionally a thin smoke test — the
per-adapter numerics are covered by the per-component integration tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import tengri.pipeline as pipeline
from tests._component_params import component_params


@pytest.mark.unit
def test_namespace_exports_protocol_and_orchestrator():
    """The pipeline namespace re-exports the contract + orchestrator helpers."""
    assert pipeline.SEDComponent is not None
    assert pipeline.ForwardState is not None
    assert pipeline.run_components is not None
    assert pipeline.merge_declared_parameters is not None
    assert pipeline.slice_params_for_component is not None
    assert pipeline.BARE_NAME_ALLOWLIST == ("redshift",)


@pytest.mark.unit
def test_namespace_exports_all_five_adapters():
    """Phase II-1 adapter cohort is reachable from the public namespace.

    Dust IR *emission* is no longer part of this top-level adapter cohort — it
    is authored as :class:`~tengri.components.sed_model_component.SEDModelComponent`
    emission components selected via the model grammar (#871)."""
    for cls_name in (
        "RadioSEDComponent",
        "IGMSEDComponent",
        "XRaySEDComponent",
        "DustAttenuationSEDComponent",
        "NebularSEDComponent",
    ):
        cls = getattr(pipeline, cls_name)
        assert cls is not None, cls_name
        assert isinstance(cls(), pipeline.SEDComponent)


@pytest.mark.unit
def test_full_five_adapter_chain_via_public_api():
    """User-perspective chain: build, merge, run."""
    chain = [
        pipeline.RadioSEDComponent(),
        pipeline.DustAttenuationSEDComponent(),
        pipeline.NebularSEDComponent(),
        pipeline.XRaySEDComponent(),
        pipeline.IGMSEDComponent(),
    ]

    merged = pipeline.merge_declared_parameters(chain)

    # Every merged name obeys the prefix or bare-name rule.
    for name in merged:
        assert (
            any(name.startswith(c.parameter_prefix) for c in chain)
            or name in pipeline.BARE_NAME_ALLOWLIST
        ), name

    # Concrete parameter values for a single forward pass.
    # Seed from the chain's own declarations — the same set ``merged`` above is
    # asserted against — then pin the values this pass depends on. Hand-rolling
    # the whole dict is what broke when the X-ray offsets were wired (#1832).
    params = {
        **component_params(*chain),
        "redshift": 1.5,
        # radio
        "radio_q_ir": 2.64,
        "radio_alpha_sf": 0.8,
        "radio_loudness": 0.0,
        "radio_alpha_agn": 0.7,
        "radio_T_e": 1e4,
        "radio_alpha_ff": -0.1,
        # dust attenuation + emission
        "dust_tau_v": 0.4,
        "dust_T": 30.0,
        "dust_beta_ir": 1.8,
        # xray
        "xray_log_nh": 20.0,
        "xray_gamma_hmxb": 2.0,
        "xray_gamma_lmxb": 1.6,
        "xray_gamma_agn": 1.8,
        "xray_E_cut": 300.0,
        "xray_delta_alpha_ox": -1.4,
        # igm
        "igm_z_mid": 7.0,
        "igm_dz": 0.5,
        "igm_log_nhi": 20.0,
    }

    wave = jnp.logspace(2, 8, 1024)
    state = pipeline.ForwardState(
        wave=wave,
        sed_intrinsic=jnp.ones_like(wave) * 1e30,
        sed_observed=jnp.ones_like(wave),
        derived={"log_mstar": 10.0, "sfr": 1.0, "L_agn_bol": 0.0},
    )

    final = pipeline.run_components(chain, state, params)

    # Every published derived key from the chain should be present. Dust IR
    # *emission* is no longer part of this adapter cohort (it is an
    # SEDModelComponent; #871), so ``sed_dust_ir`` is not expected here —
    # the attenuator still publishes ``L_ir`` for a downstream emission component.
    for key in (
        "L_ir",
        "dust_attenuation_factor",
        "sed_xray",
        "sed_nebular",
    ):
        assert key in final.derived, key

    # Attenuation slot was written.
    assert final.sed_attenuated is not None
    # IGM applied to observed-frame.
    assert final.sed_observed is not None
