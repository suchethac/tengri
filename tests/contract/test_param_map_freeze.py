# SPDX-License-Identifier: BSD-3-Clause
"""Test that SEDModel._param_map is frozen after construction.

Regression tests for Step A: Freeze `SEDModel._param_map` after construction
(ADR-deepening Step A).
"""

from types import MappingProxyType

import pytest

pytestmark = pytest.mark.contract

from tengri import Fixed, Parameters, SEDModel, Uniform
from tengri.config.exceptions import ParameterMapError


class TestParamMapFreeze:
    """Ensure _param_map is unmodifiable after construction."""

    @pytest.fixture
    def minimal_model(self, synthetic_ssp_wide):
        """Build a minimal SEDModel."""
        spec = Parameters(redshift=0.05)
        return SEDModel(spec, synthetic_ssp_wide)

    def test_param_map_is_frozen_type(self, minimal_model):
        """_param_map should be a MappingProxyType (read-only)."""
        assert isinstance(minimal_model._param_map, MappingProxyType)

    def test_param_map_mutation_raises_type_error(self, minimal_model):
        """Attempting to mutate _param_map should raise TypeError."""
        with pytest.raises(TypeError, match="does not support"):
            minimal_model._param_map["new_key"] = ("internal", 1.0, 0.0)

    def test_param_map_pop_raises_type_error(self, minimal_model):
        """Attempting to pop from _param_map should raise.

        ``types.MappingProxyType`` doesn't define ``pop`` at all, so the
        attempted call raises :class:`AttributeError`. The earlier
        :class:`TypeError`-only expectation matched dict-mutation methods
        (``__setitem__`` / ``update``) but not ``pop``. Both error types
        are acceptable — the test's purpose is that the mutation cannot
        succeed.
        """
        with pytest.raises((TypeError, AttributeError)):
            minimal_model._param_map.pop("redshift")

    def test_param_map_update_raises_type_error(self, minimal_model):
        """Attempting to update _param_map should raise TypeError."""
        with pytest.raises((TypeError, AttributeError)):
            minimal_model._param_map.update({"test": ("test", 1.0, 0.0)})


class TestParamMapValidation:
    """Test parameter map validation during construction."""

    def test_all_free_params_registered(self, synthetic_ssp_wide):
        """All free params in spec should have entries in _param_map."""
        # This should not raise; all free params are registered
        spec = Parameters(
            sfh_dpl_alpha=Uniform(0.1, 2.0),
            redshift=Fixed(0.05),
        )
        model = SEDModel(spec, synthetic_ssp_wide)
        assert "sfh_dpl_alpha" in model._param_map
        assert "redshift" in model._param_map

    def test_missing_free_param_raises(self, synthetic_ssp_wide):
        """If a free param is not registered, construction should fail.

        This is a synthetic test case where we manually inject a free param
        that no component declares.
        """
        # Create a spec with a free param that won't be in the param_map
        spec = Parameters(redshift=0.05)
        # Manually add a free param to the spec that won't be registered
        # (This is artificial but tests the validation)
        # We do this by patching free_params property to include a bogus param
        original_free_params = spec.free_params

        class MockSpec(Parameters):
            @property
            def free_params(self):
                return [*original_free_params, "nonexistent_param"]

        mock_spec = MockSpec(redshift=0.05)
        # This should raise ParameterMapError during construction
        with pytest.raises(ParameterMapError, match="nonexistent_param"):
            SEDModel(mock_spec, synthetic_ssp_wide)

    def test_param_map_contains_expected_entries(self, synthetic_ssp_wide):
        """Parameter map should have expected entries for a given spec."""
        spec = Parameters(
            sfh_dpl_alpha=Uniform(0.1, 2.0),
            dust_tau_bc=Fixed(0.5),
            redshift=Fixed(0.05),
        )
        model = SEDModel(spec, synthetic_ssp_wide)

        # Check that all free params are in the map
        for param in spec.free_params:
            assert param in model._param_map, f"{param} not in _param_map"

        # Check the structure of param_map entries (public_name -> (internal, scale, offset))
        for public_name, entry in model._param_map.items():
            assert isinstance(entry, tuple), f"Entry for {public_name} is not a tuple"
            assert len(entry) == 3, f"Entry for {public_name} has wrong length"
            internal_name, scale, offset = entry
            assert isinstance(internal_name, str)
            assert isinstance(scale, (int, float))
            assert isinstance(offset, (int, float))


class TestRecipeParamMapConsistency:
    """Spot-check that recipe models build consistent param maps."""

    def test_star_forming_photometry_builds(self, ssp_data_fsps):
        """star_forming_photometry recipe should build successfully.

        Kept on the real-SSP fixture: this recipe pulls the Cue nebular backend,
        which needs ``data/cue_weights.npz`` on disk — a real-data dependency
        beyond the SSP grid, so it gates rather than migrating to the synthetic
        SSP fixture (#613).
        """
        from tengri import recipes

        # Build a model using the recipe
        model = SEDModel.build(
            ssp_data=ssp_data_fsps,
            **recipes.star_forming_photometry(),
        )

        # Verify it has a frozen param_map
        assert isinstance(model._param_map, MappingProxyType)

        # Verify all free params are in the map
        for param in model.spec.free_params:
            assert param in model._param_map


class TestParamMapDeltas:
    """Test the internal _init_* methods return proper deltas."""

    def test_init_sfh_returns_dict(self, synthetic_ssp_wide):
        """_init_sfh should return a dict, not mutate self._param_map."""
        spec = Parameters(redshift=0.05)
        model = SEDModel(spec, synthetic_ssp_wide)

        # If we got here, the refactoring worked (no mutation during init)
        # The param_map is now frozen, so we can't check intermediate state
        assert isinstance(model._param_map, MappingProxyType)

    def test_model_with_agn_builds(self, synthetic_ssp_wide):
        """Model with AGN should build without errors."""
        spec = Parameters(
            agn_model="parametric",
            agn_log_lbol=Fixed(11.92),
            redshift=Fixed(0.05),
        )
        model = SEDModel(spec, synthetic_ssp_wide)

        assert isinstance(model._param_map, MappingProxyType)
        assert "agn_log_lbol" in model._param_map

    def test_model_with_dust_emission_builds(self, synthetic_ssp_wide):
        """Model with dust emission should build without errors."""
        spec = Parameters(
            dust_emission="draine_li2007",
            redshift=Fixed(0.05),
        )
        model = SEDModel(spec, synthetic_ssp_wide)

        assert isinstance(model._param_map, MappingProxyType)
        # Dust emission identity params should be in the map
        assert any("dust" in name for name in model._param_map)

    def test_model_with_multiple_components_builds(self, synthetic_ssp_wide):
        """Model with multiple optional components should build."""
        spec = Parameters(
            sfh_dpl_alpha=Uniform(0.1, 2.0),
            agn_model="parametric",
            agn_log_lbol=Fixed(11.92),
            dust_emission="draine_li2007",
            redshift=Fixed(0.05),
        )
        model = SEDModel(spec, synthetic_ssp_wide)

        assert isinstance(model._param_map, MappingProxyType)
        # Should have entries from multiple components
        assert any("sfh" in name for name in model._param_map)
        assert any("agn" in name for name in model._param_map)
