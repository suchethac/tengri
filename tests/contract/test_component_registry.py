# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the component registry (Phase II-6 extensibility).

Verifies:
1. Registry entries are correctly registered at module load time
2. Activation predicates work
3. Components can be retrieved from the registry
4. Grid arrays can be extracted without closure capture
"""

from __future__ import annotations

import pytest

import tengri.components.agn.skirtor_precompute

pytestmark = pytest.mark.contract
# Import component modules to trigger registration
import tengri.components.dust.dust_emission_precompute  # noqa: F401
from tengri.forward._component_registry import (
    REGISTRY,
    ComponentSpec,
    get_component,
    list_components,
    register,
)


class TestComponentRegistry:
    """Test basic registry functionality."""

    def test_dust_ir_components_registered(self):
        """Dust IR components should be registered at module load time."""
        dust_comps = list_components(family="dust_ir")
        # Required core: every model that has a precompute registration in
        # src/tengri/components/dust/dust_emission_precompute.py.
        required = {
            "dust_ir:draine_li2007",
            "dust_ir:dale2014",
            "dust_ir:astrodust",
            "dust_ir:themis",
            "dust_ir:bosa",
            "dust_ir:draine2021_pah",
        }
        assert required.issubset(set(dust_comps.keys()))

    def test_skirtor_component_registered(self):
        """SKIRTOR component should be registered at module load time."""
        agn_comps = list_components(family="agn")
        assert "agn:skirtor" in agn_comps

    def test_get_component_by_name(self):
        """Can retrieve a component by exact name."""
        spec = get_component("dust_ir:dale2014")
        assert spec is not None
        assert spec.name == "dust_ir:dale2014"

    def test_get_component_not_found(self):
        """Retrieving a nonexistent component returns None."""
        spec = get_component("nonexistent:component")
        assert spec is None

    def test_list_components_all(self):
        """List all registered components."""
        all_comps = list_components()
        assert len(all_comps) >= 7  # At least 6 dust + 1 SKIRTOR

    def test_list_components_filtered(self):
        """List only components in a specific family."""
        dust = list_components(family="dust_ir")
        assert all(name.startswith("dust_ir:") for name in dust)
        agn = list_components(family="agn")
        assert all(name.startswith("agn:") for name in agn)


class TestComponentActivation:
    """Test component activation predicates."""

    def test_dust_ir_activation(self):
        """Dust IR components should activate based on _dust_emission_model."""

        # Mock Parameters object with _dust_emission_model attribute
        class MockParams:
            _dust_emission_model = "dale2014"

        spec = get_component("dust_ir:dale2014")
        assert spec is not None
        params = MockParams()
        assert spec.activation(spec, params) is True

        # Different model should not activate
        params._dust_emission_model = "draine_li2007"
        assert spec.activation(spec, params) is False

    def test_skirtor_activation(self):
        """SKIRTOR should activate based on _agn_model."""

        class MockParams:
            _agn_model = "skirtor"

        spec = get_component("agn:skirtor")
        assert spec is not None
        params = MockParams()
        assert spec.activation(spec, params) is True

        params._agn_model = "kubota_done_full"
        assert spec.activation(spec, params) is False


class TestComponentSpecs:
    """Test ComponentSpec structure and behavior."""

    def test_component_spec_immutable(self):
        """ComponentSpec should be frozen (immutable)."""
        spec = ComponentSpec(
            name="test:component",
            precompute=lambda **kwargs: None,
            extract_arrays=lambda x: (),
            build_lookup=lambda precomp, **kwargs: lambda: None,
            apply_signature=(),
            activation=lambda spec, params: True,
        )
        with pytest.raises(AttributeError):
            spec.name = "modified"

    def test_component_spec_required_fields(self):
        """ComponentSpec requires all fields."""
        # Missing 'activation' should raise TypeError
        with pytest.raises(TypeError):
            ComponentSpec(
                name="test",
                precompute=lambda **kwargs: None,
                extract_arrays=lambda x: (),
                build_lookup=lambda precomp, **kwargs: None,
                apply_signature=(),
                # Missing activation
            )

    def test_register_duplicate_raises(self):
        """Registering a duplicate name should raise ValueError."""
        from tengri.forward._component_registry import REGISTRY

        test_spec = ComponentSpec(
            name="test:duplicate",
            precompute=lambda **kwargs: None,
            extract_arrays=lambda x: (),
            build_lookup=lambda precomp, **kwargs: None,
            apply_signature=(),
            activation=lambda spec, params: True,
        )
        # First registration should succeed
        REGISTRY.pop("test:duplicate", None)  # Clean up
        register(test_spec)

        # Second registration should fail
        with pytest.raises(ValueError, match="Duplicate component registration"):
            register(test_spec)

        # Clean up
        REGISTRY.pop("test:duplicate", None)

    def test_register_returns_spec(self):
        """register() should return the spec for use as a decorator."""
        test_spec = ComponentSpec(
            name="test:return",
            precompute=lambda **kwargs: None,
            extract_arrays=lambda x: (),
            build_lookup=lambda precomp, **kwargs: None,
            apply_signature=(),
            activation=lambda spec, params: True,
        )
        REGISTRY.pop("test:return", None)  # Clean up
        result = register(test_spec)
        assert result is test_spec
        REGISTRY.pop("test:return", None)  # Clean up


class TestDustIRSignatures:
    """Test apply_signature ordering for dust IR components."""

    def test_dale2014_signature(self):
        """Dale2014 should have 1 axis."""
        spec = get_component("dust_ir:dale2014")
        assert spec.apply_signature == ("dust_alpha_dale",)

    def test_dl07_signature(self):
        """DL07 should have 3 axes."""
        spec = get_component("dust_ir:draine_li2007")
        assert spec.apply_signature == ("dust_umin", "dust_gamma_dl", "dust_qpah")

    def test_dl14_signature(self):
        """DL14 should have 4 axes."""
        spec = get_component("dust_ir:draine_li2014")
        expected = ("dust_umin", "dust_gamma_dl", "dust_qpah", "dust_alpha_dl14")
        assert spec.apply_signature == expected


class TestSKIRTORSignature:
    """Test apply_signature for SKIRTOR."""

    def test_skirtor_signature(self):
        """SKIRTOR should have 5 axes."""
        spec = get_component("agn:skirtor")
        assert spec.apply_signature == (
            "agn_tau_skirtor",
            "agn_p_skirtor",
            "agn_q_skirtor",
            "agn_oa_skirtor",
            "agn_cos_inc",
        )
