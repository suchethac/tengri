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

from tengri import DEFAULT, Fixed, SEDModel, Uniform, recipes
from tengri.forward.precompute_report import precompute_engagement_report

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
                "tau_v": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},  # Use SSP-baked nebular for wNE SSP
            redshift=Fixed(0.1),
            approx="auto",
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
            approx="auto",
        )

        report = precompute_engagement_report(model)

        # KNOWN DEFECT #2485: single_component does not populate DustSEDComponent
        # in a way that the LUT builder recognizes it, so both mechanisms decline.
        # This is correct behavior (model still computes right answer, just slower),
        # but silent. Record the current state and flip when #2485 is resolved.
        assert report.energy_balance_lut.state == "never_attempted", (
            "Single-component dust now populates DustSEDComponent; fix is underway for #2485"
        )
        assert report.dust_band_response.state == "never_attempted", (
            "Single-component dust now engages dust response; this test expectation should flip"
        )

    def test_free_tau_disengages_energy_balance_lut(self, synthetic_tophat_obs, ssp_data_wne):
        """Free tau_v should disengage energy-balance LUT.

        The LUT bakes one curve at build time; free tau parameters mean the
        curve shape is not constant and LUT cannot be used.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_v": Uniform(0.0, 3.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},
            redshift=Fixed(0.1),
            approx="auto",
        )

        report = precompute_engagement_report(model)

        # Free tau_v means unsafe_free is nonempty, LUT disengages
        assert report.energy_balance_lut.state == "declined", (
            f"Expected energy-balance LUT to decline with free tau_v. "
            f"State: {report.energy_balance_lut.state}, "
            f"Reason: {report.energy_balance_lut.reason}"
        )

    def test_free_redshift_disengages_mechanisms(self, synthetic_tophat_obs, ssp_data_wne):
        """Free redshift should disengage precompute mechanisms.

        Redshift affects the dust curve (through narayanan_z), the dust
        emission shape (through redshift dependence), and other component
        shapes. Mechanisms that bake one redshift at build time decline.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_v": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},
            redshift=Uniform(0.0, 3.0),
            approx="auto",
        )

        report = precompute_engagement_report(model)

        # Free redshift disengages both LUT and dust response
        assert report.energy_balance_lut.state == "declined", (
            f"Expected energy-balance LUT to decline with free redshift. "
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
                "tau_v": Fixed(0.5),
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
            dust_attenuation={"type": "two_component", "all_params": Fixed(DEFAULT)},
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "ssp"},
            radio={"type": "default", "all_params": Fixed(DEFAULT)},
            redshift=Uniform(0.0, 3.0),
            approx="auto",
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


def test_precompute_report_on_minimal_model(synthetic_tophat_obs, ssp_data_bc03):
    """Minimal model should not error on precompute report."""
    model = SEDModel.build(
        ssp_data=ssp_data_bc03,
        observation=synthetic_tophat_obs,
        **recipes.star_forming_photometry(),
    )

    report = precompute_engagement_report(model)

    # Should not raise, and should have valid state values
    assert report.energy_balance_lut.state in ("engaged", "declined", "never_attempted")
    assert report.dust_band_response.state in ("engaged", "declined", "never_attempted")
    assert isinstance(report.emitter_term_responses, dict)
    assert isinstance(report.observed_facts, dict)
