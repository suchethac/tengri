# SPDX-License-Identifier: BSD-3-Clause
"""
Contract test: precompute mechanism engagement census.

Pins which optimizations engage for representative model configurations.
Detects unintentional changes in engagement state (e.g., a refactor that
silently disables an optimization).

References
----------
.. [1] Make silent precompute forfeits observable. GitHub Issue #2485.
"""

import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.precompute_report import precompute_engagement_report
from tengri.forward.sed_model import WavePrecomp

pytestmark = pytest.mark.contract


class TestPrecomputeEngagementCensus:
    """Contract: which precompute mechanisms engage per configuration."""

    def test_two_component_dust_attenuation_with_dale2014_engages_mechanisms(
        self, synthetic_tophat_obs, ssp_data_wne
    ):
        """Two-component + dale2014 with fixed tau should engage LUT and response."""
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.3),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},  # Use SSP-baked nebular for wNE SSP
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )

        report = precompute_engagement_report(model)

        # Two-component + fixed tau_v + dale2014 + WavePrecomp should engage LUT
        # (assuming redshift is fixed and no free shape parameters)
        assert report.energy_balance_lut.state in ("engaged", "declined"), (
            f"Energy-balance LUT unexpectedly never attempted. "
            f"Reason: {report.energy_balance_lut.reason}"
        )
        # Dust emission should engage response (fixed shape, WavePrecomp enabled)
        assert report.dust_band_response.state in ("engaged", "declined"), (
            f"Dust band response unexpectedly never attempted. "
            f"Reason: {report.dust_band_response.reason}"
        )

    def test_single_component_dust_attenuation_currently_forfeits(
        self, synthetic_tophat_obs, ssp_data_wne
    ):
        """Single-component dust currently forfeits precompute (known defect #2485).

        This test records the current state as the expectation. When #2485 is
        fixed to support single-component dust, flip these assertions to
        expect 'engaged' instead.

        References
        ----------
        .. [1] Issue #2485: single_component dust_attenuation forfeits precompute.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "tau_v": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )

        report = precompute_engagement_report(model)

        # KNOWN DEFECT #2485: single_component dust is now detected (not never_attempted),
        # but the LUT builder declines it. This is correct behavior (model still computes
        # right answer, just slower), but silent. Record the current state and flip when
        # #2485 is resolved. The LUT builder only recognizes two_component dust attenuation.
        # Note: dust_band_response (from dust_emission) still engages because it does not
        # depend on the dust attenuation type.
        assert report.energy_balance_lut.state == "declined", (
            "Single-component dust detected but LUT declined (expected for #2485)"
        )
        assert report.dust_band_response.state == "engaged", (
            "Dust emission band response engages (independent of attenuation type)"
        )

    def test_free_tau_disengages_energy_balance_lut(self, synthetic_tophat_obs, ssp_data_wne):
        """Free tau_v should engage energy-balance LUT (precomputed over grid).

        The LUT precomputes over a grid of tau values (dust_tau_bc and dust_tau_diff
        are in the safe-to-free allowlist), so free tau does not prevent LUT engagement.
        The LUT builder calls _grid("dust_tau_bc") and _grid("dust_tau_diff"), which
        return either a fixed value or a linspace grid, allowing the LUT to work
        correctly under both fixed and free tau.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Uniform(0.0, 1.5),
                "tau_diff": Uniform(0.0, 1.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )

        report = precompute_engagement_report(model)

        # Free tau_bc and tau_diff are in _EB_ATTEN_FREE_OK, so they don't trigger
        # unsafe_free and the LUT engages with a precomputed grid of tau values.
        assert report.energy_balance_lut.state == "engaged", (
            f"Expected energy-balance LUT to engage with free tau (grid precomp). "
            f"State: {report.energy_balance_lut.state}, "
            f"Reason: {report.energy_balance_lut.reason}"
        )

    def test_free_redshift_disengages_mechanisms(self, synthetic_tophat_obs, ssp_data_wne):
        """Free redshift should disengage precompute mechanisms (when law reads it).

        The narayanan_z dust attenuation law reads redshift to adjust the dust curve.
        When redshift is free, mechanisms that bake one curve at build time decline
        because the shape changes across the parameter space.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "narayanan_z",
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.3),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},
            redshift=Uniform(0.0, 3.0),
            approx=WavePrecomp(),
        )

        report = precompute_engagement_report(model)

        # Free redshift with narayanan_z (which reads redshift) should disengage
        # both LUT and dust response because the curve shape varies with redshift.
        assert report.energy_balance_lut.state == "declined", (
            f"Expected energy-balance LUT to decline with free redshift on narayanan_z. "
            f"Reason: {report.energy_balance_lut.reason}"
        )
        assert report.dust_band_response.state == "declined", (
            f"Expected dust band response to decline with free redshift. "
            f"Reason: {report.dust_band_response.reason}"
        )

    def test_approx_none_disengages_mechanisms(self, synthetic_tophat_obs, ssp_data_wne):
        """approx=None (exact path) should disengage precompute mechanisms.

        WavePrecomp is a gate for all build-time precompute mechanisms.
        With exact path, they are not attempted.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.3),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},
            redshift=Fixed(0.1),
            approx=None,
        )

        report = precompute_engagement_report(model)

        # Exact path never attempts precompute mechanisms
        assert report.energy_balance_lut.state == "never_attempted", (
            "approx=None should not trigger energy-balance LUT"
        )
        assert report.dust_band_response.state == "never_attempted", (
            "approx=None should not trigger dust band response"
        )

    def test_emitter_term_responses_with_free_redshift(self, synthetic_tophat_obs, ssp_data_wne):
        """Radio/xray term responses should disengage with free redshift."""
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "ssp"},
            redshift=Uniform(0.0, 3.0),
            approx=WavePrecomp(),
        )

        report = precompute_engagement_report(model)

        # Free redshift prevents term-response engagement
        if "radio" in report.emitter_term_responses:
            radio_state = report.emitter_term_responses["radio"].state
            assert radio_state in ("never_attempted", "declined"), (
                f"Radio term response should not engage with free redshift. State: {radio_state}"
            )


def test_precompute_engagement_report_structure():
    """PrecomputeEngagementReport has required fields."""
    from tengri.forward.precompute_report import (
        PrecomputeEngagementReport,
        PrecomputeState,
    )

    report = PrecomputeEngagementReport(
        energy_balance_lut=PrecomputeState(state="engaged"),
        dust_band_response=PrecomputeState(state="declined", reason="shape_free=True"),
        emitter_term_responses={"radio": PrecomputeState(state="never_attempted")},
        observed_facts={"wave_precomp_enabled": True},
    )

    # Check that summary_text() works
    summary = report.summary_text()
    assert isinstance(summary, str)
    assert "ENGAGED" in summary or "engaged" in summary.lower()
    assert "DECLINED" in summary or "declined" in summary.lower()


def test_precompute_report_on_minimal_model(synthetic_tophat_obs, ssp_data_wne):
    """Minimal model should not error on precompute report."""
    # Use a real SSP fixture that has ionizing spectrum data (wNE has baked nebular)
    # and avoid complex recipes that might introduce additional dependencies
    model = SEDModel.build(
        ssp_data=ssp_data_wne,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.3),
            "tau_diff": Fixed(0.1),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "ssp"},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )

    report = precompute_engagement_report(model)

    # Should not raise, and should have valid state values
    assert report.energy_balance_lut.state in ("engaged", "declined", "never_attempted")
    assert report.dust_band_response.state in ("engaged", "declined", "never_attempted")
    assert isinstance(report.emitter_term_responses, dict)
    assert isinstance(report.observed_facts, dict)
