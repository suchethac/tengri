# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for recipe-level introspection via recipe_parameters().

Frozen: the exact parameter records (name, owner, free/fixed status) returned
by recipe_parameters() for each canonical recipe, without requiring SSP data.
Tests the mapping between nested-dict grammar (recipes) and ParameterRecord lists.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri import parse_groups, recipes
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.registry import ParameterRecord, recipe_parameters
from tengri.parameters.sentinels import FIXED, FREE


class TestRecipeParametersStructure:
    """Test that recipe_parameters returns sorted, unique ParameterRecord lists."""

    def test_star_forming_photometry_returns_sorted_records(self):
        """Returned records are sorted by name and have no duplicates."""
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)

        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, ParameterRecord) for p in params)

        names = [p.name for p in params]
        assert names == sorted(names), "Records must be sorted by name"
        assert len(names) == len(set(names)), "Records must be unique"

    def test_quiescent_z0_returns_sorted_records(self):
        """Quiescent recipe also returns sorted, unique records."""
        recipe = recipes.quiescent_z0()
        params = recipe_parameters(recipe)

        names = [p.name for p in params]
        assert names == sorted(names)
        assert len(names) == len(set(names))


class TestRecipeParametersFreeVsFixed:
    """Test free_only parameter filter distinguishes free and fixed params."""

    def test_quiescent_z0_has_fixed_redshift(self):
        """quiescent_z0 has redshift=Fixed(0.05): free_only excludes it."""
        recipe = recipes.quiescent_z0()

        params_free = recipe_parameters(recipe, free_only=True)
        names_free = [p.name for p in params_free]
        assert "redshift" not in names_free, "Fixed redshift must not appear with free_only=True"

        params_all = recipe_parameters(recipe, free_only=False)
        names_all = [p.name for p in params_all]
        assert "redshift" in names_all, "Fixed redshift must appear with free_only=False"

    def test_mock_recovery_minimal_has_fixed_redshift(self):
        """mock_recovery_minimal has redshift fixed, excluded from free_only."""
        recipe = recipes.mock_recovery_minimal()
        params_free = recipe_parameters(recipe, free_only=True)
        names_free = [p.name for p in params_free]
        assert "redshift" not in names_free

    def test_star_forming_has_free_redshift(self):
        """star_forming_photometry has redshift free."""
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)
        names = [p.name for p in params]
        assert "redshift" in names


class TestRecipeParametersConsistency:
    """Test consistency between recipe_parameters and Parameters."""

    def test_matches_parameters_free_params(self):
        """Free params from recipe_parameters match parse_groups.free_params (subset)."""
        recipe = recipes.star_forming_photometry()
        params = parse_groups(**recipe)
        recipe_params = recipe_parameters(recipe, free_only=True)
        recipe_names = {p.name for p in recipe_params}
        expected_names = set(params.free_params)

        # Note: SFH parameters are not yet in the registry (ADR-0005 gap #3),
        # so we expect recipe_parameters to be a subset
        assert recipe_names.issubset(expected_names)
        assert any(n.startswith("dust_") or n == "redshift" for n in recipe_names)

    def test_matches_parameters_all_params(self):
        """All params from recipe_parameters match parse_groups.all_params (subset)."""
        recipe = recipes.mock_recovery_minimal()
        params = parse_groups(**recipe)
        recipe_params = recipe_parameters(recipe, free_only=False)
        recipe_names = {p.name for p in recipe_params}
        expected_names = set(params.all_params)

        # SFH parameters not yet in registry, so subset
        assert recipe_names.issubset(expected_names)
        assert any(n.startswith("dust_") for n in recipe_names)


class TestRecipeParametersManualRecipes:
    """Test with manually constructed recipe dicts."""

    def test_minimal_recipe_returns_records(self):
        """Minimal recipe with FREE/FIXED sentinels works correctly."""
        recipe = {
            "sfh": {"type": "dpl", "*": FREE},
            "dust": {"law": "power_law", "type": "two_component", "*": FIXED},
            "neb": {"type": "none"},
            "redshift": Fixed(0.05),
        }
        params = recipe_parameters(recipe)

        assert len(params) > 0
        names = [p.name for p in params]
        # Redshift is fixed, so shouldn't appear in free_only=True
        assert "redshift" not in names

    def test_recipe_with_explicit_priors(self):
        """Recipe with explicit Uniform priors for free params."""
        recipe = {
            "sfh": {"type": "dpl", "*": FREE},
            "dust": {
                "law_diff": "calzetti",
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
        """Every ParameterRecord has a non-empty owner from tengri."""
        recipe = recipes.star_forming_photometry()
        params = recipe_parameters(recipe)

        assert all(p.owner for p in params)
        assert all(p.owner.startswith("tengri.") for p in params)


class TestRecipeParametersErrors:
    """Test error handling for malformed recipes."""

    def test_invalid_group_raises_error(self):
        """Unknown group key raises ValueError."""
        recipe = {"invalid_group": {"type": "dpl"}}
        with pytest.raises(ValueError, match="Unknown group key"):
            recipe_parameters(recipe)

    def test_invalid_sfh_type_raises_error(self):
        """Unknown SFH type raises ValueError."""
        recipe = {"sfh": {"type": "nonexistent_type"}}
        with pytest.raises(ValueError, match="Unknown SFH type"):
            recipe_parameters(recipe)

    def test_invalid_dust_type_raises_error(self):
        """Unknown dust type raises ValueError."""
        recipe = {
            "sfh": {"type": "dpl"},
            "dust": {"type": "invalid_dust_type"},
        }
        with pytest.raises(ValueError, match="Unknown dust type"):
            recipe_parameters(recipe)
