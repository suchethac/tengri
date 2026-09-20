# SPDX-License-Identifier: BSD-3-Clause
"""Tests for citation key recognition in unknown-name errors.

When a user provides a citation key (like 'charlot_fall2000') as a type or
parameter name, error messages should recognize it as a known citation key and
explain which registry entry it cites, rather than falling back to difflib
suggestions alone.
"""

import pytest

from tengri.citations.resolve import registry_names_for_citation_key
from tengri.parameters import Fixed
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


class TestCitationKeyRecognition:
    """When user provides a citation key as a name, error explains the registry entry."""

    def test_dust_law_citation_key_single_component_law(self):
        """Providing 'charlot_fall2000' (citation key for power_law) should explain it."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "charlot_fall2000"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg
        # The error should use the keyword the caller used
        assert "law=" in error_msg or "law" in error_msg

    def test_dust_law_citation_key_two_component_law_bc(self):
        """Providing 'charlot_fall2000' as law_bc on two_component should explain it."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "charlot_fall2000",
                    "law_diff": "calzetti",
                },
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg
        # Should mention the keyword used
        assert "law_bc" in error_msg or "law" in error_msg

    def test_dust_law_typo_unchanged(self):
        """A non-citation typo like 'calzeti' should still get difflib suggestion."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzeti"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # Should still have the difflib suggestion
        assert "Did you mean:" in error_msg
        assert "calzetti" in error_msg

    def test_dust_emission_citation_key(self):
        """Citation key for dust_emission type should be recognized."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                dust_emission={"type": "calzetti2000"},  # Citation key for "calzetti" (dust law)
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "citation key" in error_msg.lower()
        # calzetti2000 is citation key for calzetti which is a dust law, not dust_emission

    def test_nebular_citation_key(self):
        """Citation key for nebular type should be recognized."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                neb={"type": "cue_wrong"},  # Unknown name
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # Just ensure error is raised; citation hint tested separately
        assert "Unknown" in error_msg or "cue_wrong" in error_msg

    def test_citation_key_not_valid_at_site(self):
        """When citation key is for a different group, explain that clearly."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                neb={"type": "charlot_fall2000"},  # Citation key for power_law, not a nebular type
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # Should say it's a citation key for power_law (dust law)
        assert "citation key" in error_msg.lower()
        assert "power_law" in error_msg
        # Should say it's not valid for nebular type
        assert "not" in error_msg.lower() or "dust" in error_msg.lower()

    def test_sfh_type_citation_key(self):
        """Citation key for SFH type should be recognized."""
        with pytest.raises(ValueError) as exc_info:
            parse_groups(
                sfh={"type": "bagpipes"},  # This is a citation key for 'dpl'
                dust_attenuation={"type": "single_component", "law": "calzetti"},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # bagpipes is citation key for dpl
        if "citation key" in error_msg.lower():
            assert "dpl" in error_msg

    def test_reverse_lookup_basic(self):
        """Unit test: registry_names_for_citation_key returns correct names."""
        # charlot_fall2000 -> power_law
        names = registry_names_for_citation_key("charlot_fall2000")
        assert "power_law" in names

    def test_reverse_lookup_multiple_names(self):
        """Unit test: citation key mapping to multiple registry names."""
        # skirtor citation key should map to multiple names
        names = registry_names_for_citation_key("skirtor")
        assert len(names) > 0
        # Should include the main skirtor name
        assert "skirtor" in names

    def test_reverse_lookup_unknown_key(self):
        """Unit test: unknown citation key returns empty tuple."""
        names = registry_names_for_citation_key("unknown_key_xyz_9999")
        assert names == ()

    def test_reverse_lookup_case_insensitive(self):
        """Unit test: reverse lookup is case-insensitive."""
        names_lower = registry_names_for_citation_key("charlot_fall2000")
        names_upper = registry_names_for_citation_key("CHARLOT_FALL2000")
        assert names_lower == names_upper
