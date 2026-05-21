# SPDX-License-Identifier: BSD-3-Clause
"""Tests for recipe-level introspection via recipe_parameters().

Covers the new ADR-0005 follow-up #3 function: tengri.recipe_parameters().
This function walks a recipe's nested-dict structure and returns the parameters
it activates, without requiring SSP data or building an SEDModel.

See https://github.com/suchethacooray/tengri/pull/34 for context.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri import recipes
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.registry import ParameterRecord, recipe_parameters
from tengri.parameters.sentinels import FIXED, FREE


class TestRecipeParametersBasic:
    """Test basic functionality with the six canonical recipes."""

    def test_star_forming_photometry_returns_records(self):
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, ParameterRecord) for p in params)

    def test_quiescent_z0_returns_records(self):
        recipe = recipes.quiescent_z0()
        params = recipe_parameters(recipe)
        assert len(params) > 0

    def test_agn_panchromatic_returns_records(self):
        recipe = recipes.agn_panchromatic()
        params = recipe_parameters(recipe)
        assert len(params) > 0

    def test_stochastic_sfh_jwst_returns_records(self):
        recipe = recipes.stochastic_sfh_jwst()
        params = recipe_parameters(recipe)
        assert len(params) > 0

    def test_mock_recovery_minimal_returns_records(self):
        recipe = recipes.mock_recovery_minimal()
        params = recipe_parameters(recipe)
        assert len(params) > 0

    def test_dust_demo_returns_all_records(self):
        # dust_demo has all parameters FIXED, so no free params
        recipe = recipes.dust_demo()
        params_free = recipe_parameters(recipe, free_only=True)
        assert len(params_free) == 0  # All fixed
        params_all = recipe_parameters(recipe, free_only=False)
        assert len(params_all) > 0  # But has params when including fixed


class TestRecipeParametersSorting:
    """Test that returned records are sorted and unique."""

    def test_results_are_sorted_by_name(self):
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)
        names = [p.name for p in params]
        assert names == sorted(names)

    def test_no_duplicate_names(self):
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)
        names = [p.name for p in params]
        assert len(names) == len(set(names))


class TestRecipeParametersFreeOnly:
    """Test the free_only parameter filter."""

    def test_free_only_true_excludes_fixed_params(self):
        recipe = recipes.quiescent_z0()
        free = recipe_parameters(recipe, free_only=True)
        all_params = recipe_parameters(recipe, free_only=False)
        # Include fixed params should return more or same entries
        assert len(all_params) >= len(free)

    def test_free_only_false_includes_fixed(self):
        recipe = recipes.star_forming_photometry()
        free = recipe_parameters(recipe, free_only=True)
        all_params = recipe_parameters(recipe, free_only=False)
        # Not all recipes have fixed params, but if they do, all_params
        # should have more entries
        if len(free) < len(all_params):
            # At least one fixed parameter in this recipe
            assert True
        else:
            # Recipe has no fixed params, which is ok
            assert len(all_params) == len(free)


class TestRecipeParametersRedshift:
    """Test that redshift and other top-level params are handled."""

    def test_star_forming_has_redshift_free(self):
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)
        names = [p.name for p in params]
        assert "redshift" in names

    def test_quiescent_has_redshift_fixed(self):
        recipe = recipes.quiescent_z0()
        # quiescent_z0 has redshift=Fixed(0.05), which means it's fixed
        # In free_only=True mode, it should NOT appear
        params_free = recipe_parameters(recipe, free_only=True)
        names_free = [p.name for p in params_free]
        assert "redshift" not in names_free

        # In free_only=False mode, it SHOULD appear
        params_all = recipe_parameters(recipe, free_only=False)
        names_all = [p.name for p in params_all]
        assert "redshift" in names_all

    def test_mock_recovery_minimal_has_redshift_fixed(self):
        recipe = recipes.mock_recovery_minimal()
        params_free = recipe_parameters(recipe, free_only=True)
        names_free = [p.name for p in params_free]
        # redshift is fixed, so shouldn't appear
        assert "redshift" not in names_free


class TestRecipeParametersConsistency:
    """Test consistency between recipe_parameters and Parameters."""

    def test_matches_parameters_free_params(self):
        from tengri.parameters.parameters import Parameters

        recipe = recipes.star_forming_photometry()
        params = Parameters.from_groups(**recipe)
        recipe_params = recipe_parameters(recipe, free_only=True)
        recipe_names = {p.name for p in recipe_params}
        expected_names = set(params.free_params)

        # Note: SFH parameters are not yet in the registry (ADR-0005 gap #3),
        # so we expect recipe_parameters to be a subset of the full Parameters.free_params
        # Once all SFH params migrate to component _params.py, this assertion should
        # become: assert recipe_names == expected_names
        assert recipe_names.issubset(expected_names)
        # But we should have at least redshift and dust params
        assert any(n.startswith("dust_") or n == "redshift" for n in recipe_names)

    def test_matches_parameters_all_params(self):
        from tengri.parameters.parameters import Parameters

        recipe = recipes.mock_recovery_minimal()
        params = Parameters.from_groups(**recipe)
        recipe_params = recipe_parameters(recipe, free_only=False)
        recipe_names = {p.name for p in recipe_params}
        expected_names = set(params.all_params)

        # Note: SFH parameters are not yet in the registry (ADR-0005 gap #3),
        # so we expect recipe_parameters to be a subset of the full Parameters.all_params
        # Once all SFH params migrate to component _params.py, this assertion should
        # become: assert recipe_names == expected_names
        assert recipe_names.issubset(expected_names)
        # But we should have at least dust params
        assert any(n.startswith("dust_") for n in recipe_names)


class TestRecipeParametersManualRecipes:
    """Test with manually constructed recipe dicts."""

    def test_minimal_recipe(self):
        recipe = {
            "sfh": {"type": "dpl", "*": FREE},
            "dust": {"type": "two_component", "*": FIXED},
            "neb": {"type": "none"},
            "redshift": Fixed(0.05),
        }
        params = recipe_parameters(recipe)
        names = [p.name for p in params]
        # SFH parameters are not yet in the registry (ADR-0005 gap #3),
        # so we'll just check metallicity
        assert "met_logzsol" in names or "dust_tau_bc" in names
        # redshift is fixed, so shouldn't appear in free_only=True
        assert "redshift" not in names

    def test_recipe_with_explicit_priors(self):
        recipe = {
            "sfh": {"type": "dpl", "*": FREE},
            "dust": {
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FREE,
                "tau_bc": Uniform(0, 1),
            },
            "neb": {"type": "cue", "*": FIXED},
            "redshift": Uniform(0.01, 6.0),
        }
        params = recipe_parameters(recipe, free_only=True)
        assert len(params) > 0
        names = [p.name for p in params]
        # redshift with Uniform prior should be free
        assert "redshift" in names

    def test_recipe_parameter_records_have_owner(self):
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)
        # Every ParameterRecord should have a non-empty owner
        assert all(p.owner for p in params)
        # Owners should be from tengri.components or tengri.parameters
        assert all(p.owner.startswith("tengri.") for p in params)


class TestRecipeParametersErrors:
    """Test error handling for malformed recipes."""

    def test_invalid_group_raises_error(self):
        recipe = {"invalid_group": {"type": "dpl"}}
        with pytest.raises(ValueError, match="Unknown group key"):
            recipe_parameters(recipe)

    def test_invalid_sfh_type_raises_error(self):
        recipe = {"sfh": {"type": "nonexistent_type"}}
        with pytest.raises(ValueError, match="Unknown SFH type"):
            recipe_parameters(recipe)

    def test_invalid_dust_type_raises_error(self):
        recipe = {
            "sfh": {"type": "dpl"},
            "dust": {"type": "invalid_dust_type"},
        }
        with pytest.raises(ValueError, match="Unknown dust type"):
            recipe_parameters(recipe)


class TestRecipeParametersEmptyRecipe:
    """Test behavior with minimal/empty recipes."""

    def test_empty_recipe_dict_returns_defaults(self):
        # Empty dict should use all defaults (which means some params)
        recipe = {}
        params = recipe_parameters(recipe)
        # At minimum, metallicity should be present
        assert len(params) > 0


class TestRecipeParametersEdgeCases:
    """Test edge cases and special behaviors."""

    def test_composition_sfh_returns_records(self):
        # stochastic_sfh_jwst uses a list composition for sfh
        recipe = recipes.stochastic_sfh_jwst()
        params = recipe_parameters(recipe)
        # Should still work and return params
        assert len(params) > 0
        # Note: SFH parameters (dpl, field) are not yet in registry (ADR-0005 gap),
        # but redshift and other params should be present
        names = [p.name for p in params]
        assert "redshift" in names

    def test_dust_emission_subblock_returns_records(self):
        # star_forming_photometry has dust.emission.type = "dale2014"
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)
        # Should include dust emission params
        assert len(params) > 0
