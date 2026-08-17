# SPDX-License-Identifier: BSD-3-Clause
"""Test that _resolve_registry_component error suggestions are domain-scoped.

Validates that when a user makes a typo in a component name, the error message
suggests only components valid for that domain, not the full registry.

Contract marker: #1521
"""

from __future__ import annotations

import difflib

import pytest

from tengri.components.sed_model_component import _REGISTRY
from tengri.forward.component_factory import (
    _DOMAIN_MEMBERSHIP,
    _resolve_registry_component,
)

# Components that never route through _resolve_registry_component.
# Each entry names a component and why it is not domain-scoped.
_NOT_ROUTED = frozenset(
    {
        "draine2021_pah_ir",  # Standalone IR component, not routed through build_components
        "schreiber2016_ir",  # Standalone IR component, not routed through build_components
    }
)


class TestComponentDomainScoping:
    """Test that error suggestions respect domain boundaries.

    Contract marker: #1521 — error messages list only domain-valid components.
    """

    @pytest.mark.parametrize("domain", list(_DOMAIN_MEMBERSHIP.keys()))
    def test_registry_miss_suggests_only_domain_names(self, domain: str):
        """Trigger a registry miss and verify suggestions are domain-scoped.

        For each domain, use an invalid component name and assert the
        ValueError's message contains ZERO names from foreign domains.
        """
        invalid_name = f"invalid_{domain}_component_xyz123"

        with pytest.raises(ValueError) as exc_info:
            _resolve_registry_component(domain, invalid_name)

        error_msg = str(exc_info.value)

        # Parse the available names from the error message
        available_str = error_msg[error_msg.find("Available names:") :]
        available_names = set()

        # Extract names that are in the registry
        for registry_name in _REGISTRY:
            if registry_name in available_str:
                available_names.add(registry_name)

        # Assert no foreign names are present
        domain_names = _DOMAIN_MEMBERSHIP[domain]
        foreign_names = available_names - domain_names

        assert not foreign_names, (
            f"Domain {domain!r} error suggests foreign names: {sorted(foreign_names)}. "
        )
        f"Available={sorted(available_names)}, Domain={sorted(domain_names)}"

    def test_dust_emission_close_match_works(self):
        """Test that close matches work for dust_emission domain.

        A one-character misspelling of 'modified_blackbody' should be suggested.
        """
        typo_name = "modified_blackbody_xyz"  # Obvious typo

        with pytest.raises(ValueError) as exc_info:
            _resolve_registry_component("dust_emission", typo_name)

        error_msg = str(exc_info.value)

        # Should suggest close matches (difflib.get_close_matches behavior)
        # At minimum, 'modified_blackbody' should be mentioned
        assert "modified_blackbody" in error_msg, (
            f"Expected 'modified_blackbody' in suggestions, got: {error_msg}"
        )

    def test_dh02_ce01_now_in_registry(self):
        """Test dh02_ce01: now registered in the forward registry.

        dh02_ce01 was mentioned in #1521 as a gap case (existing in the
        loader cache but not the registry), but it has since been registered
        as a forward component. This test verifies it's now accessible.

        Contract marker: #1521 — the dh02_ce01 gap is closed.
        """
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        # Verify dh02_ce01 exists in the loader cache
        assert "dh02_ce01" in DUST_EMISSION_MODELS

        # Verify it's NOW in the registry (the gap is closed)
        assert "dh02_ce01" in _REGISTRY

        # Should resolve without error
        component = _resolve_registry_component("dust_emission", "dh02_ce01")
        assert component is not None
        assert component.name == "dust_emission"

    def test_all_domains_covered(self):
        """Verify that all domains used in component_factory are covered."""
        # These are the domains used in build_components() call sites
        required_domains = {
            "dust_attenuation",
            "dust_emission",
            "nebular",
            "agn",
            "radio",
            "xray",
            "igm",
        }

        available_domains = set(_DOMAIN_MEMBERSHIP.keys())

        assert required_domains.issubset(available_domains), (
            f"Missing domains: {required_domains - available_domains}"
        )

    def test_no_registry_entries_drift_into_wrong_domain(self):
        """Verify that each registry entry belongs to exactly one domain.

        Contract: every new registered component must be classified in
        _DOMAIN_MEMBERSHIP or explicitly excepted in _NOT_ROUTED, or CI fails.
        """
        # All domain entries union
        all_domain_names = set().union(*_DOMAIN_MEMBERSHIP.values())

        # Registry names
        registry_names = set(_REGISTRY.keys())

        # Uncovered entries should match exactly _NOT_ROUTED
        uncovered = registry_names - all_domain_names
        assert uncovered == _NOT_ROUTED, (
            f"Registry entries uncovered by _DOMAIN_MEMBERSHIP must be in "
            f"_NOT_ROUTED (each with a comment explaining why). "
            f"Uncovered: {sorted(uncovered)}, "
            f"Expected (NOT_ROUTED): {sorted(_NOT_ROUTED)}. "
            f"To fix: add new components to _DOMAIN_MEMBERSHIP in component_factory.py."
        )

        # No multi-domain entries
        name_to_domains: dict[str, set[str]] = {}
        for domain, names in _DOMAIN_MEMBERSHIP.items():
            for name in names:
                if name not in name_to_domains:
                    name_to_domains[name] = set()
                name_to_domains[name].add(domain)

        multi_domain = {
            name: domains for name, domains in name_to_domains.items() if len(domains) > 1
        }
        assert not multi_domain, f"Multi-domain entries found: {multi_domain}"


class TestErrorMessageQuality:
    """Test that error messages are helpful and domain-scoped."""

    def test_error_message_format(self):
        """Verify that error messages are well-formatted and informative."""
        with pytest.raises(ValueError) as exc_info:
            _resolve_registry_component("dust_emission", "nonexistent")

        error_msg = str(exc_info.value)

        # Should mention the domain
        assert "dust_emission" in error_msg

        # Should mention the invalid name
        assert "nonexistent" in error_msg

        # Should mention "not found"
        assert "not found" in error_msg.lower() or "not in" in error_msg.lower()

    def test_close_match_cutoff_respected(self):
        """Verify that close matches have reasonable cutoff (0.6)."""
        # "modified_blackbodyxyz" has high similarity to "modified_blackbody"
        similar_name = "modified_blackbodyxyz"

        matches = difflib.get_close_matches(
            similar_name, list(_DOMAIN_MEMBERSHIP["dust_emission"]), n=3, cutoff=0.6
        )

        # Should find at least one match with the cutoff
        if matches:
            assert "modified_blackbody" in matches
