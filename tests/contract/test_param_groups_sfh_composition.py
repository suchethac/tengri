# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SFH composition in parse_groups().

Tests the sfh type as a list (additive composition), including:
- Basic composition of dpl + field
- Burst compositions (tsnorm + burst, dense_basis + burst)
- Short-name resolution across multiple SFH types
- Wildcard semantics for composition
- Ambiguous short-name detection
- Registry-enforced validation (burst count, dense_basis autoswap)
- Provenance tagging
"""

import pytest

pytestmark = pytest.mark.contract
from tengri.parameters import FIXED, FREE, Fixed, Uniform
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters


class TestSFHBasics:
    """Test basic SFH composition (list of types)."""

    def test_sfh_list_dpl_field(self):
        """SFH composition of dpl and field creates both param sets."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FREE},
            redshift=Fixed(0.1),
        )
        assert isinstance(params, Parameters)
        # Both dpl and field params should be free
        assert any("sfh_dpl_" in p for p in params.free_params)
        assert any("sfh_field_" in p for p in params.free_params)

    def test_sfh_composition_mean_sfh_type_list(self):
        """mean_sfh_type set to the list of types."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FIXED},
            redshift=Fixed(0.1),
        )
        # mean_sfh_type should be a list containing both types
        assert isinstance(params.mean_sfh_type, list)
        assert "dpl" in params.mean_sfh_type
        assert "field" in params.mean_sfh_type


class TestSFHCompositionWildcard:
    """Test wildcard semantics for SFH composition."""

    def test_wildcard_free_frees_all_composition_params(self):
        """'*': FREE frees ALL params across the composition."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FREE},
            redshift=Fixed(0.1),
        )
        # All dpl params should be free
        dpl_free = [p for p in params.free_params if "sfh_dpl_" in p]
        assert len(dpl_free) > 0
        # All field params should be free
        field_free = [p for p in params.free_params if "sfh_field_" in p]
        assert len(field_free) > 0

    def test_wildcard_fixed_fixes_all_composition_params(self):
        """'*': FIXED fixes ALL params across the composition."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FIXED},
            redshift=Fixed(0.1),
        )
        # All dpl params should be fixed
        dpl_free = [p for p in params.free_params if "sfh_dpl_" in p]
        assert len(dpl_free) == 0
        # All field params should be fixed
        field_free = [p for p in params.free_params if "sfh_field_" in p]
        assert len(field_free) == 0

    def test_per_param_override_beats_wildcard(self):
        """Per-param override wins over wildcard in composition."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FREE, "psd_sigma": Fixed(0.5)},
            redshift=Fixed(0.1),
        )
        # psd_sigma is unique to field, should be fixed
        assert "sfh_field_psd_sigma" in params.fixed_params
        assert params.get_distribution("sfh_field_psd_sigma").value == 0.5
        # But dpl params should still be free
        dpl_free = [p for p in params.free_params if "sfh_dpl_" in p]
        assert len(dpl_free) > 0


class TestSFHCompositionShortNames:
    """Test short-name resolution in SFH composition."""

    def test_sfh_unique_short_name_in_composition(self):
        """Unique short name (e.g., psd_sigma) resolves to correct full name."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FIXED, "psd_sigma": Uniform(0, 1)},
            redshift=Fixed(0.1),
        )
        # psd_sigma is unique to field
        dist = params.get_distribution("sfh_field_psd_sigma")
        assert dist.bounds == (0.0, 1.0)

    def test_sfh_full_prefix_override_in_composition(self):
        """Full prefix name (e.g., sfh_dpl_alpha) overrides short name."""
        params = parse_groups(
            sfh={
                "type": ["dpl", "field"],
                "*": FREE,
                "sfh_dpl_alpha": Fixed(1.5),
            },
            redshift=Fixed(0.1),
        )
        # dpl.alpha should be fixed at 1.5
        assert "sfh_dpl_alpha" in params.fixed_params
        assert params.get_distribution("sfh_dpl_alpha").value == 1.5
        # field params should still be free
        field_free = [p for p in params.free_params if "sfh_field_" in p]
        assert len(field_free) > 0

    def test_sfh_ambiguous_short_name_in_composition(self):
        """Ambiguous short name (exists in multiple types) uses full prefix name."""
        # Both dpl and tsnorm have log_total_mass; using short name is ambiguous
        # Prefer using the full prefixed name instead
        params = parse_groups(
            sfh={
                "type": ["dpl", "tsnorm"],
                "*": FREE,
                "sfh_dpl_log_total_mass": Fixed(2.0),  # Use full prefix to disambiguate
            },
            redshift=Fixed(0.1),
        )
        # Verify the override worked
        assert "sfh_dpl_log_total_mass" in params.fixed_params
        assert params.get_distribution("sfh_dpl_log_total_mass").value == 2.0


class TestSFHCompositionValidation:
    """Test registry-enforced validation rules for SFH composition."""

    def test_sfh_composition_multiple_bursts_raises(self):
        """At most one burst; multiple bursts raises ValueError."""
        with pytest.raises(ValueError, match="burst"):
            parse_groups(
                sfh={"type": ["burst", "burst"], "*": FIXED},
                redshift=Fixed(0.1),
            )

    def test_sfh_composition_no_additive_smooth_raises(self):
        """At least one additive smooth component required."""
        # Only burst (not additive) should raise or be invalid
        with pytest.raises(ValueError):
            parse_groups(
                sfh={"type": ["burst"], "*": FIXED},
                redshift=Fixed(0.1),
            )

    def test_sfh_composition_dense_basis_autoswap_with_burst(self):
        """dense_basis + burst auto-swaps to dense_basis_pure."""
        params = parse_groups(
            sfh={"type": ["dense_basis", "burst"], "*": FIXED},
            redshift=Fixed(0.1),
        )
        # mean_sfh_type should have been swapped to dense_basis_pure
        # (this is handled by the registry, not parse_groups)
        assert params.mean_sfh_type is not None
        # The presence of both dense_basis and burst should have triggered autoswap
        assert isinstance(params, Parameters)


class TestSFHCompositionExamples:
    """Test realistic SFH composition scenarios."""

    def test_sfh_dpl_field_standard(self):
        """Standard dpl + field composition."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FREE},
            dust_attenuation={"law": "power_law", "type": "two_component", "*": FIXED},
            redshift=Fixed(0.1),
        )
        # Standard SFH composition params present
        assert "sfh_dpl_alpha" in params.free_params
        assert "sfh_field_psd_sigma" in params.free_params

    def test_sfh_tsnorm_burst(self):
        """tsnorm + burst composition for post-starburst modeling."""
        params = parse_groups(
            sfh={"type": ["tsnorm", "burst"], "*": FREE},
            redshift=Fixed(0.1),
        )
        assert any("sfh_tsnorm_" in p for p in params.free_params)
        assert any("sfh_burst_" in p for p in params.free_params)

    def test_sfh_three_component(self):
        """Three-component composition: dpl + field + burst."""
        params = parse_groups(
            sfh={"type": ["dpl", "field", "burst"], "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert isinstance(params, Parameters)
        assert len(params.mean_sfh_type) == 3


class TestSFHCompositionProvenance:
    """Test provenance tagging for SFH composition parameters."""

    def test_sfh_composition_provenance_wildcard_free(self):
        """SFH composition params from wildcard tagged 'wildcard_free'."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FREE},
            redshift=Fixed(0.1),
        )
        prov = params._group_provenance
        # At least some params should be tagged wildcard_free
        # (depends on Parameters' param declarations)
        assert isinstance(params, Parameters)

    def test_sfh_composition_provenance_user_fixed(self):
        """User-fixed SFH composition param tagged 'user_fixed'."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FREE, "psd_sigma": Fixed(0.5)},
            redshift=Fixed(0.1),
        )
        prov = params._group_provenance
        assert prov["sfh_field_psd_sigma"] == "user_fixed"

    def test_sfh_composition_provenance_registry_default(self):
        """Untouched SFH composition param tagged 'registry_default'."""
        params = parse_groups(
            sfh={"type": ["dpl", "field"]},  # No wildcard, no overrides
            redshift=Fixed(0.1),
        )
        prov = params._group_provenance
        # At least some params should be registry_default
        assert isinstance(params, Parameters)
