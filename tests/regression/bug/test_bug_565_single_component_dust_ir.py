# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #565: single_component dust IR re-emission.

Bug: ``dust={'type': 'single_component', 'emission': {...}}`` silently dropped
the energy-balanced dust IR re-emission (far-IR ~370x too small vs the
equivalent ``two_component`` build). The single-screen attenuator
(``DustAttenuationSEDComponent``) correctly published ``state.derived['L_ir']``,
but two things were missing: (i) the factory never appended a
``DustEmissionSEDComponent`` to re-radiate it, and (ii) even once appended, the
emission adapter declared no inputs, so the topological sort ran it *before* the
attenuator — it read the ``L_ir`` fallback of 0.0 and emitted nothing.

Fix: ``component_factory.build_components`` appends ``DustEmissionSEDComponent``
in the single_component branch (analytic ``modified_blackbody`` natively; loud
error for template-grid libraries), and ``DustEmissionSEDComponent`` declares
``L_ir`` as an optional input so the orchestrator orders it after the attenuator.
"""

import warnings

import numpy as np
import pytest

try:
    import tengri
except ImportError:
    tengri = None

pytestmark = pytest.mark.regression_bug

# Rest-frame far-IR window covering the modified-blackbody peak (~100 um for
# typical dust_T ~ 30 K). 5e5 A = 50 um.
_FAR_IR_AA = 5e5


def _far_ir_sum(state):
    """Sum of rest-frame L_nu over the far-IR window [erg/s/Hz]."""
    wave = np.asarray(state.wave)
    sed = np.asarray(state.sed_intrinsic)
    return float(sed[wave > _FAR_IR_AA].sum())


def _build(dust):
    # Hoist nested 'emission' to separate dust_emission top-level group
    dust_attenuation_config = dict(dust)
    dust_emission_config = dust_attenuation_config.pop("emission", None)

    kwargs = {
        "ssp_data": tengri.load_ssp(),
        "sfh": {
            "type": "delayed",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "log_total_mass": 10.0,
        },
        "dust_attenuation": dust_attenuation_config,
        "redshift": tengri.Fixed(0.05),
    }
    if dust_emission_config is not None:
        kwargs["dust_emission"] = dust_emission_config

    return tengri.SEDModel.build(**kwargs)


@pytest.mark.skipif(tengri is None, reason="tengri not installed")
def test_single_component_emission_reradiates_absorbed_energy():
    """single_component + MBB emission must re-radiate the absorbed energy.

    Far-IR luminosity must be vastly larger with emission than without — the
    #565 bug left it at the bare attenuated-stellar tail (~370x too small).
    """
    try:
        tengri.load_ssp()
    except (FileNotFoundError, OSError):
        pytest.skip("SSP data not available (expected in CI)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_em = _build(
            {
                "law": "power_law",
                "type": "single_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_v": tengri.Fixed(2.0),
                "emission": {
                    "type": "modified_blackbody",
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                },
            }
        )
        no_em = _build(
            {
                "law": "power_law",
                "type": "single_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_v": tengri.Fixed(2.0),
            }
        )

    fir_with = _far_ir_sum(with_em.predict_state({}))
    fir_without = _far_ir_sum(no_em.predict_state({}))
    ratio = fir_with / max(fir_without, 1e-50)
    assert ratio > 50, (
        f"single_component far-IR ratio with/without emission = {ratio:.1f}, "
        f"expected >> 1. Dust IR is not being re-radiated (#565)."
    )


@pytest.mark.skipif(tengri is None, reason="tengri not installed")
def test_single_component_emission_matches_two_component(real_ssp_only):
    """The single-screen IR must be energy-balance-consistent with two_component.

    With matched diffuse optical depth, the absorbed luminosity (hence the
    re-radiated far-IR) should agree to within a small factor; #565 made the
    single-screen value ~370x too small.

    Needs the real grid: the balance is set by UV/optical absorption, which the
    smooth synthetic #613 SSP does not reproduce (two-component far-IR ~0 there,
    so the ratio diverges). ``real_ssp_only`` skips on synthetic-only CI.
    """
    try:
        tengri.load_ssp()
    except (FileNotFoundError, OSError):
        pytest.skip("SSP data not available (expected in CI)")

    mbb = {"type": "modified_blackbody", "all_params": tengri.Fixed(tengri.DEFAULT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        single = _build(
            {
                "law": "power_law",
                "type": "single_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_v": tengri.Fixed(2.0),
                "emission": mbb,
            }
        )
        two = _build(
            {
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": tengri.Fixed(2.0),
                "emission": mbb,
            }
        )

    fir_single = _far_ir_sum(single.predict_state({}))
    fir_two = _far_ir_sum(two.predict_state({}))
    rel = fir_single / max(fir_two, 1e-50)
    assert 0.5 < rel < 2.0, (
        f"single_component far-IR / two_component far-IR = {rel:.3f}, expected ~1 "
        f"(matched diffuse optical depth). Single-screen energy balance is off (#565)."
    )


@pytest.mark.skipif(tengri is None, reason="tengri not installed")
def test_single_component_publishes_nonzero_l_ir():
    """The attenuator must publish a healthy L_ir for the emission to consume."""
    try:
        tengri.load_ssp()
    except (FileNotFoundError, OSError):
        pytest.skip("SSP data not available (expected in CI)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(
            {
                "law": "power_law",
                "type": "single_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_v": tengri.Fixed(2.0),
                "emission": {
                    "type": "modified_blackbody",
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                },
            }
        )
    L_ir = float(np.asarray(model.predict_state({}).derived.get("L_ir", 0.0)))
    assert L_ir > 1e42, f"Published L_ir = {L_ir:.2e} erg/s, expected > 1e42 (#565)."


@pytest.mark.skipif(tengri is None, reason="tengri not installed")
def test_single_component_grid_emission_reradiates():
    """Grid IR libraries now work on the single-screen path too (was #565).

    Pre-migration, ``single_component`` + a grid IR template (dale2014) raised —
    the fused-kernel single-screen path never wired grid IR. The unified
    SEDModelComponent dispatch (ADR-0019) wires every emission component uniformly:
    the attenuator publishes ``L_ir`` and the component re-radiates it regardless of
    single- vs two-component screen. So the grid template now re-radiates the
    absorbed energy instead of failing loud.
    """
    try:
        tengri.load_ssp()
    except (FileNotFoundError, OSError):
        pytest.skip("SSP data not available (expected in CI)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_em = _build(
            {
                "law": "power_law",
                "type": "single_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_v": tengri.Fixed(2.0),
                "emission": {"type": "dale2014", "all_params": tengri.Fixed(tengri.DEFAULT)},
            }
        )
        no_em = _build(
            {
                "law": "power_law",
                "type": "single_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_v": tengri.Fixed(2.0),
            }
        )

    ratio = _far_ir_sum(with_em.predict_state({})) / max(
        _far_ir_sum(no_em.predict_state({})), 1e-50
    )
    assert ratio > 50, (
        f"single_component + grid dale2014 far-IR ratio = {ratio:.1f}, expected >> 1: "
        "the unified component dispatch must re-radiate grid IR on the single-screen path (#565)."
    )
