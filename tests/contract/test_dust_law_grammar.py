# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust attenuation law grammar and explicit requirements.

This module tests the dust law grammar rework that makes attenuation laws
explicit and requires them to be specified. Tests cover:
- Missing law errors with helpful messages
- single_component law requirements and grammar
- two_component law requirements and pair form
- wg00 unchanged
- builders enforcement
- round-trip consistency
"""

import pytest

from tengri.parameters import FIXED, Fixed
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract


class TestSingleComponentMissingLaw:
    """single_component requires 'law' in the grammar."""

    def test_missing_law_raises_error(self):
        """Missing law on single_component should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "tau_v": Fixed(0.1)},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "law" in error_msg.lower()
        assert "calzetti" in error_msg or "power_law" in error_msg

    def test_law_bc_on_single_component_errors(self):
        """Using law_bc on single_component should raise ValueError naming 'law'."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "single_component",
                    "law_bc": "calzetti",
                    "tau_v": Fixed(0.1),
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "law" in error_msg

    def test_law_diff_on_single_component_errors(self):
        """Using law_diff on single_component should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "single_component",
                    "law_diff": "calzetti",
                    "tau_v": Fixed(0.1),
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "single_component" in error_msg or "law" in error_msg


class TestSingleComponentValidLaw:
    """single_component accepts 'law' and applies it."""

    def test_law_key_works(self):
        """Using law key on single_component should work."""
        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.1)},
            redshift=Fixed(0.1),
        )
        assert params.dust_model == "single_component"
        assert params.dust_law_bc == "calzetti"
        # For single component, dust_law_diff should also be set to the same
        assert params.dust_law_diff == "calzetti"

    def test_law_key_power_law(self):
        """Using law='power_law' on single_component should work."""
        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={"type": "single_component", "law": "power_law", "tau_v": Fixed(0.1)},
            redshift=Fixed(0.1),
        )
        assert params.dust_law_bc == "power_law"
        assert params.dust_law_diff == "power_law"

    def test_unknown_law_raises_with_suggestion(self):
        """An unknown law name should raise with a close-match suggestion."""
        with pytest.raises(ValueError, match="Unknown dust law"):
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "single_component",
                    "law": "calzeti",
                    "tau_v": Fixed(0.1),
                },
                redshift=Fixed(0.1),
            )


class TestTwoComponentMissingLaw:
    """two_component requires 'law' OR both 'law_bc' and 'law_diff'."""

    def test_missing_all_laws_raises_error(self):
        """Missing all law keys on two_component should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "tau_bc": Fixed(0.1),
                    "tau_diff": Fixed(0.2),
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "law" in error_msg.lower()
        assert "calzetti" in error_msg or "power_law" in error_msg

    def test_only_law_bc_raises_error(self):
        """Missing law_diff when law_bc is given should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "calzetti",
                    "tau_bc": Fixed(0.1),
                    "tau_diff": Fixed(0.2),
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "both" in error_msg.lower() or "law=" in error_msg

    def test_only_law_diff_raises_error(self):
        """Missing law_bc when law_diff is given should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law_diff": "calzetti",
                    "tau_bc": Fixed(0.1),
                    "tau_diff": Fixed(0.2),
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "both" in error_msg.lower() or "law=" in error_msg

    def test_law_and_law_bc_together_raises_error(self):
        """Specifying both 'law' and 'law_bc' is ambiguous."""
        with pytest.raises(ValueError, match="ambiguous"):
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "law_bc": "smc",
                    "tau_bc": Fixed(0.1),
                    "tau_diff": Fixed(0.2),
                },
                redshift=Fixed(0.1),
            )


class TestTwoComponentValidLaw:
    """two_component accepts 'law' OR 'law_bc' + 'law_diff' pair."""

    def test_law_key_sets_both_screens(self):
        """Using law key on two_component should set both law_bc and law_diff."""
        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.1),
                "tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.1),
        )
        assert params.dust_law_bc == "calzetti"
        assert params.dust_law_diff == "calzetti"

    def test_law_bc_law_diff_pair_works(self):
        """Using law_bc and law_diff pair on two_component should work."""
        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "tau_bc": Fixed(0.1),
                "tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.1),
        )
        assert params.dust_law_bc == "calzetti"
        assert params.dust_law_diff == "power_law"

    def test_unknown_law_bc_raises_with_suggestion(self):
        with pytest.raises(ValueError, match="Unknown dust law"):
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "calzeti",
                    "law_diff": "power_law",
                    "tau_bc": Fixed(0.1),
                    "tau_diff": Fixed(0.2),
                },
                redshift=Fixed(0.1),
            )


class TestWG00Unaffected:
    """wg00 dust type does not use the law/law_bc/law_diff grammar."""

    def test_wg00_builds_without_law_keys(self):
        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "wg00",
                "dust_curve": "mw",
                "geometry": "shell",
                "structure": "homogeneous",
                "tau_v": Fixed(0.3),
            },
            redshift=Fixed(0.1),
        )
        assert params.dust_model == "wg00"


class TestParametersFlatKwargs:
    """The flat Parameters(...) kwarg interface.

    Parameters(...) is the expert escape hatch (bypasses the grammar/build
    path entirely) and keeps its long-standing power_law default when no law
    kwarg is given at all, for backward compatibility with the large body of
    existing direct-Parameters() tests. Explicit law kwargs (``dust_law_bc``/
    ``dust_law_diff`` for two_component, ``dust_law_bc`` for single_component)
    still work and override the default -- the grammar path (SEDModel.build /
    parse_groups) is where explicit laws are actually enforced (see
    TestSingleComponentMissingLaw / TestTwoComponentMissingLaw above).
    """

    def test_single_component_dust_law_bc_kwarg_works(self):
        spec = Parameters(
            dust_model="single_component",
            dust_law_bc="calzetti",
            dust_tau_v=Fixed(0.3),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "calzetti"

    def test_two_component_dust_law_bc_diff_pair_kwargs_work(self):
        spec = Parameters(
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_law_diff="power_law",
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "power_law"

    def test_dust_off_skips_law_requirement(self):
        """dust_model='off' (no dust) must not demand a law."""
        spec = Parameters(
            dust_model="off",
            apply_igm=False,
        )
        assert spec.dust_model == "off"

    def test_wg00_skips_law_requirement(self):
        spec = Parameters(
            dust_model="wg00",
            dust_wg00_curve="mw",
            dust_wg00_geometry="shell",
            dust_wg00_structure="homogeneous",
            dust_tau_v=Fixed(0.3),
            apply_igm=False,
        )
        assert spec.dust_model == "wg00"


class TestBuildersEnforceLaw:
    """The builders.dust.* factories enforce the same grammar."""

    def test_single_component_builder_requires_law(self):
        from tengri import builders

        with pytest.raises((TypeError, ValueError)):
            builders.dust.single_component()

    def test_single_component_builder_law_works(self):
        from tengri import builders

        out = builders.dust.single_component(law="calzetti")
        assert out["law"] == "calzetti"

    def test_two_component_builder_requires_law_or_pair(self):
        from tengri import builders

        with pytest.raises((TypeError, ValueError)):
            builders.dust.two_component()

    def test_two_component_builder_law_works(self):
        from tengri import builders

        out = builders.dust.two_component(law="calzetti")
        assert out["law"] == "calzetti"

    def test_two_component_builder_pair_works(self):
        from tengri import builders

        out = builders.dust.two_component(law_bc="calzetti", law_diff="power_law")
        assert out["law_bc"] == "calzetti"
        assert out["law_diff"] == "power_law"


class TestRoundTrip:
    """Test round-trip consistency: parse_groups -> parameters_to_groups."""

    def test_single_component_law_roundtrip(self):
        """single_component with law should round-trip correctly."""
        from tengri.parameters.groups import parameters_to_groups

        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.1)},
            redshift=Fixed(0.1),
        )
        groups = parameters_to_groups(params)
        assert groups["dust_attenuation"]["law"] == "calzetti"
        assert "law_bc" not in groups["dust_attenuation"]

    def test_two_component_law_roundtrip(self):
        """two_component with a shared law should round-trip to 'law'."""
        from tengri.parameters.groups import parameters_to_groups

        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.1),
                "tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.1),
        )
        groups = parameters_to_groups(params)
        assert groups["dust_attenuation"]["law"] == "calzetti"
        assert "law_bc" not in groups["dust_attenuation"]
        assert "law_diff" not in groups["dust_attenuation"]

    def test_two_component_pair_roundtrip(self):
        """two_component with differing screen laws should round-trip to the pair."""
        from tengri.parameters.groups import parameters_to_groups

        params = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "tau_bc": Fixed(0.1),
                "tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.1),
        )
        groups = parameters_to_groups(params)
        assert groups["dust_attenuation"]["law_bc"] == "calzetti"
        assert groups["dust_attenuation"]["law_diff"] == "power_law"
        assert "law" not in groups["dust_attenuation"]

    def test_roundtrip_reparses_identically(self):
        """The emitted groups dict must re-parse to an identical spec."""
        from tengri.parameters.groups import parameters_to_groups

        params = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "smc",
                "tau_bc": Fixed(0.4),
                "tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.1),
        )
        groups = parameters_to_groups(params)
        reparsed = parse_groups(**groups)
        assert reparsed.dust_law_bc == params.dust_law_bc
        assert reparsed.dust_law_diff == params.dust_law_diff
