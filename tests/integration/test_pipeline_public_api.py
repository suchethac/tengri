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


@pytest.mark.unit
def test_namespace_exports_protocol_and_orchestrator():
    """The pipeline namespace re-exports the contract + orchestrator helpers."""
    assert pipeline.SEDComponent is not None
    assert pipeline.ForwardState is not None
    assert pipeline.run_components is not None
    assert pipeline.merge_declared_parameters is not None
    assert pipeline.slice_params_for_component is not None
    assert pipeline.default_params_dict is not None
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

    # Concrete parameter values for a single forward pass, read off the same
    # declarations ``merge_declared_parameters`` just walked. The literal that
    # stood here spelled out twenty keys and was complete when written; it went
    # stale the day ``xray_det_hmxb`` gained a reader (#1832), which is the
    # failure a user-perspective test should be the last place to reproduce.
    params = pipeline.default_params_dict(chain, overrides={"redshift": 1.5})

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
