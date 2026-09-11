# SPDX-License-Identifier: BSD-3-Clause
"""Test flat Parameters validation of dust shape parameters and law overrides.

The flat form should raise ParameterError when a user names a dust shape parameter
(dust_Rv, dust_delta, dust_slope, dust_bump_strength) or provides a dust_law_overrides
entry that the resolved law does not read, mirroring the grammar's per-screen validation.
"""

import pytest

from tengri import DEFAULT, Fixed
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract


class TestFlatDustShapeValidation:
    """Test flat Parameters validation of dust shape parameters."""

    def test_calzetti_rejects_dust_rv(self):
        """calzetti does not read dust_Rv; explicitly passing it should raise."""
        with pytest.raises(ParameterError, match="dust_Rv"):
            Parameters(
                dust_model="single_component",
                dust_law_bc="calzetti",
                dust_tau_v=Fixed(0.3),
                dust_Rv=Fixed(4.0),
                redshift=Fixed(0.5),
            )

    def test_power_law_rejects_dust_bump_strength(self):
        """power_law does not read dust_bump_strength; explicitly passing it should raise."""
        with pytest.raises(ParameterError, match="dust_bump_strength"):
            Parameters(
                dust_model="two_component",
                dust_law_bc="power_law",
                dust_bump_strength=Fixed(1.0),
                redshift=Fixed(0.5),
            )

    def test_kriek_conroy_accepts_dust_delta(self):
        """kriek_conroy reads dust_delta; this should build successfully."""
        spec = Parameters(
            dust_model="two_component",
            dust_law_bc="kriek_conroy",
            dust_delta=Fixed(-0.3),
            redshift=Fixed(0.5),
        )
        assert spec.dust_law_bc == "kriek_conroy"

    def test_power_law_diff_accepts_dust_slope(self):
        """power_law reads dust_slope; shared (no screen) dust_slope should build."""
        spec = Parameters(
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_law_diff="power_law",
            dust_slope=Fixed(-1.0),
            redshift=Fixed(0.5),
        )
        assert spec.dust_law_diff == "power_law"

    def test_dust_law_overrides_rejects_unknown_screen(self):
        """dust_law_overrides with an unknown screen name should raise."""
        with pytest.raises(ParameterError, match="birth_cloud"):
            Parameters(
                dust_model="two_component",
                dust_law_overrides={"birth_cloud": {"dust_slope": -1.0}},
                redshift=Fixed(0.5),
            )

    def test_dust_law_overrides_rejects_param_not_read_by_law(self):
        """dust_law_overrides with a parameter not read by the law should raise."""
        with pytest.raises(ParameterError, match="dust_delta"):
            Parameters(
                dust_model="two_component",
                dust_law_bc="calzetti",
                dust_law_overrides={"bc": {"dust_delta": 0.2}},
                redshift=Fixed(0.5),
            )

    def test_dust_law_overrides_accepts_valid_param(self):
        """dust_law_overrides with a valid parameter and law should build."""
        spec = Parameters(
            dust_model="two_component",
            dust_law_bc="power_law",
            dust_law_overrides={"bc": {"dust_slope": -1.0}},
            redshift=Fixed(0.5),
        )
        assert spec.dust_law_overrides == {"bc": {"dust_slope": -1.0}}

    def test_omission_keeps_defaults(self):
        """Omitting dust parameters keeps registry defaults (escape hatch)."""
        spec = Parameters(
            dust_model="two_component",
            redshift=Fixed(0.5),
        )
        assert spec.dust_law_bc == "power_law"
        # No error should be raised

    def test_error_parity_grammar_and_flat(self):
        """Grammar and flat error messages for the same violation are equivalent."""
        # Grammar version
        grammar_error = None
        try:
            parse_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "tau_bc": Fixed(0.5),
                    "tau_diff": Fixed(0.2),
                    "delta_bc": -0.3,
                },
                redshift=Fixed(0.5),
            )
        except ParameterError as e:
            grammar_error = str(e)

        # Flat version
        flat_error = None
        try:
            Parameters(
                dust_model="two_component",
                dust_law_bc="calzetti",
                dust_law_overrides={"bc": {"dust_delta": -0.3}},
                redshift=Fixed(0.5),
            )
        except ParameterError as e:
            flat_error = str(e)

        # Both should raise ParameterError
        assert grammar_error is not None
        assert flat_error is not None
        # Both messages should contain key phrases
        assert "is not read by" in grammar_error
        assert "calzetti" in grammar_error
        assert "is not read by" in flat_error
        assert "calzetti" in flat_error

    def test_grammar_with_registry_defaults_builds(self):
        """Grammar with calzetti + tau_bc + tau_diff should build (registry defaults validated)."""
        # This case failed at collection before the fix; grammar passes registry
        # defaults for dust_bump_strength/delta/Rv/slope without validation issues
        spec = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.5),
        )
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "calzetti"

    def test_grammar_validated_flag_not_leaked(self):
        """_grammar_validated should not appear in user_provided, flat_provenance, or
        valid_param_names."""
        spec = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.5),
        )
        assert "_grammar_validated" not in spec._user_provided
        assert "_grammar_validated" not in spec._flat_provenance
        assert "_grammar_validated" not in spec._valid_param_names

    def test_explicit_bypass_flag_accepted(self):
        """Explicit _grammar_validated=True should bypass validation (internal use)."""
        # This bypasses the flat validation for internal calls
        spec = Parameters(
            dust_model="single_component",
            dust_law_bc="calzetti",
            dust_tau_v=Fixed(0.3),
            dust_Rv=Fixed(4.0),
            redshift=Fixed(0.5),
            _grammar_validated=True,
        )
        # Should build without raising, even though calzetti doesn't read dust_Rv
        assert spec.dust_law_bc == "calzetti"
