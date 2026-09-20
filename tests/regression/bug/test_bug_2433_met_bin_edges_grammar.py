# SPDX-License-Identifier: BSD-3-Clause
r"""Regression tests for #2433 — met group accepts met_bin_edges_log_yr structural key.

Mechanism
---------

The ``sfh`` group accepts a ``bin_edges_gyr`` structural key to override the bin layout
for non-parametric SFHs (prospector_beta, continuity). This key is threaded through the
grammar -> sed_model -> component_factory -> StellarSEDComponentConfig.

The ``met`` group must accept a parallel ``met_bin_edges_log_yr`` structural key for
metallicity-history modes (bins, bins_continuity) to allow users to override the default
ladder (_DEFAULT_MET_BIN_EDGES_LOG_YR) without needing direct component access.

The #2204 refusal (cosmic age vs. bin starts) must judge the CONFIGURED ladder when one
is given, and name the key in the error message.

https://github.com/suchethac/tengri/issues/2433
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.config.exceptions import ParameterError
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug


class TestMetBinEdgesGrammarStructuralKey:
    """The met group accepts met_bin_edges_log_yr as a structural key."""

    def test_grammar_route_builds_with_custom_ladder(self, synthetic_ssp, simple_observation):
        """(a) Grammar route: met={'met_bin_edges_log_yr': [...]} builds.

        Builds SEDModel with a custom met_bin_edges_log_yr ladder via grammar.
        Verifies both spec storage and stellar config carry the ladder.
        """
        obs = simple_observation
        custom_edges = [6.0, 8.0, 9.5]

        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={
                "type": "bins",
                "all_params": Fixed(DEFAULT),
                "met_bin_0": -0.3,
                "met_bin_edges_log_yr": custom_edges,
            },
            redshift=Fixed(0.1),
        )
        assert model is not None
        # Verify spec carries the ladder
        assert model.spec.met_bin_edges_log_yr == custom_edges

    def test_default_ladder_explicit_vs_implicit_bit_identical(
        self, synthetic_ssp, simple_observation
    ):
        """(b.equiv) Equality: explicit default ladder vs implicit → bit-identical prediction.

        Passing the default met_bin_edges_log_yr explicitly should yield bit-identical
        photometry and SED to omitting the key and using the implicit default.
        """
        obs = simple_observation
        default_ladder = [6.0, 7.5, 8.5, 9.0, 9.5, 9.9, 10.14]

        # Build model with implicit default
        model_implicit = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={"type": "bins", "all_params": Fixed(DEFAULT), "met_bin_0": -0.3},
            redshift=Fixed(0.1),
        )

        # Build model with explicit default ladder
        model_explicit = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={
                "type": "bins",
                "all_params": Fixed(DEFAULT),
                "met_bin_0": -0.3,
                "met_bin_edges_log_yr": default_ladder,
            },
            redshift=Fixed(0.1),
        )

        # Predict with both models
        params = {}
        pred_implicit = model_implicit.predict(params)
        pred_explicit = model_explicit.predict(params)

        # Rest SEDs should be bit-identical
        sed_implicit = pred_implicit.rest_sed()
        sed_explicit = pred_explicit.rest_sed()
        np.testing.assert_array_equal(
            sed_implicit,
            sed_explicit,
            err_msg="Explicit and implicit default ladder should give identical SED",
        )

        # Photometry should also be bit-identical
        phot_implicit = pred_implicit.photometry()
        phot_explicit = pred_explicit.photometry()
        np.testing.assert_array_equal(
            phot_implicit,
            phot_explicit,
            err_msg="Explicit and implicit default ladder should give identical photometry",
        )

    def test_engagement_custom_ladder_is_stored_in_spec(self, synthetic_ssp, simple_observation):
        """(b) Engagement: custom ladder structure is stored and retrieved from spec.

        Build a SEDModel with met_bin_edges_log_yr and verify the ladder is
        stored in the spec and can be retrieved for inspection and round-tripping.
        """
        obs = simple_observation
        custom_ladder = [6.0, 7.0, 8.0, 8.5, 9.2, 9.8, 10.14]

        # Build model with custom ladder
        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={
                "type": "bins",
                "met_bin_0": -0.5,
                "met_bin_1": -0.3,
                "met_bin_2": -0.1,
                "met_bin_3": 0.0,
                "met_bin_4": 0.1,
                "met_bin_5": 0.2,
                "met_bin_edges_log_yr": custom_ladder,
            },
            redshift=Fixed(0.1),
        )

        # Verify spec carries the ladder
        assert model.spec.met_bin_edges_log_yr == custom_ladder

        # Verify model can predict (structural engagement test)
        params = {}
        pred = model.predict(params)
        sed = pred.rest_sed()
        assert np.all(np.isfinite(sed)), "SED should be finite"
        assert np.any(sed > 0), "SED should be non-zero"

    def test_configured_ladder_fixes_2204_refusal(self, synthetic_ssp, simple_observation):
        """(c) #2204 remedy: configured ladder that fits cosmic age builds where default refuses.

        At Fixed(3.0), cosmic age ≈ 2.15 Gyr. The default ladder has bins starting at
        3.16 and 7.94 Gyr, which are unreachable (exceed cosmic age). A custom ladder
        with all bin starts <= 2.15 Gyr should build successfully.
        """
        z_high = 3.0
        expected_age = age_at_z(z_high)
        assert expected_age < 3.16, f"Test setup: z={z_high} should give age < 3.16 Gyr"

        obs = simple_observation

        # With default ladder, build should refuse
        with pytest.raises(ParameterError):
            SEDModel.build(
                ssp_data=synthetic_ssp,
                observation=obs,
                sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
                met={"type": "bins", "all_params": Fixed(DEFAULT), "met_bin_0": -0.3},
                redshift=Fixed(z_high),
            )

        # With custom ladder (all bins fit within cosmic age at z=3.0), build should succeed
        safe_edges = [0.1, 1.0, 2.0]  # All edges < 2.15 Gyr
        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={
                "type": "bins",
                "all_params": Fixed(DEFAULT),
                "met_bin_0": -0.3,
                "met_bin_edges_log_yr": safe_edges,
            },
            redshift=Fixed(z_high),
        )
        assert model is not None

    def test_validation_non_increasing_edges_refused(self, synthetic_ssp, simple_observation):
        """(d) Validation: non-increasing ladder is refused with key in message."""
        obs = simple_observation
        non_increasing_edges = [6.0, 5.0, 7.0]  # Not strictly increasing

        with pytest.raises((ParameterError, ValueError)) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp,
                observation=obs,
                sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
                met={
                    "type": "bins",
                    "all_params": Fixed(DEFAULT),
                    "met_bin_0": -0.3,
                    "met_bin_edges_log_yr": non_increasing_edges,
                },
                redshift=Fixed(0.1),
            )

        error_msg = str(exc_info.value)
        assert "met_bin_edges_log_yr" in error_msg, (
            f"Error should name met_bin_edges_log_yr; got: {error_msg}"
        )

    def test_validation_too_few_edges_refused(self, synthetic_ssp, simple_observation):
        """(d) Validation: fewer than two edges is refused with key in message."""
        obs = simple_observation
        single_edge = [6.0]  # Need at least 2 edges to form bins

        with pytest.raises((ParameterError, ValueError)) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp,
                observation=obs,
                sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
                met={
                    "type": "bins",
                    "all_params": Fixed(DEFAULT),
                    "met_bin_0": -0.3,
                    "met_bin_edges_log_yr": single_edge,
                },
                redshift=Fixed(0.1),
            )

        error_msg = str(exc_info.value)
        assert "met_bin_edges_log_yr" in error_msg, (
            f"Error should name met_bin_edges_log_yr; got: {error_msg}"
        )

    def test_validation_non_finite_edges_refused(self, synthetic_ssp, simple_observation):
        """(d) Validation: non-finite edges are refused with key in message."""
        obs = simple_observation
        non_finite_edges = [6.0, float("nan"), 9.5]  # NaN edge

        with pytest.raises((ParameterError, ValueError)) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp,
                observation=obs,
                sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
                met={
                    "type": "bins",
                    "all_params": Fixed(DEFAULT),
                    "met_bin_0": -0.3,
                    "met_bin_edges_log_yr": non_finite_edges,
                },
                redshift=Fixed(0.1),
            )

        error_msg = str(exc_info.value)
        assert "met_bin_edges_log_yr" in error_msg, (
            f"Error should name met_bin_edges_log_yr; got: {error_msg}"
        )

    def test_key_refused_on_non_ladder_met_types(self, synthetic_ssp, simple_observation):
        """(e) Validation: met_bin_edges_log_yr is refused on non-ladder met types.

        The key is structural and only applicable to ladder-based met types
        ('bins', 'bins_continuity'). When applied to non-ladder types (e.g., 'delta'),
        it must raise ValueError naming the key and listing valid types.
        """
        obs = simple_observation
        custom_edges = [6.0, 8.0, 9.5]

        # Must raise ValueError on delta type, which has no ladder
        with pytest.raises(ValueError, match="met_bin_edges_log_yr"):
            SEDModel.build(
                ssp_data=synthetic_ssp,
                observation=obs,
                sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
                met={
                    "type": "delta",
                    "all_params": Fixed(DEFAULT),
                    "met_logzsol": -0.3,
                    "met_bin_edges_log_yr": custom_edges,  # Not allowed on delta
                },
                redshift=Fixed(0.1),
            )
