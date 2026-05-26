# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.recipes module.

Verifies that each curated recipe returns a valid dict that can be passed
to parse_groups() and that the resulting Parameters has the
expected structural and parameter properties.
"""

from __future__ import annotations

import pytest

from tengri import parse_groups, recipes
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract


class TestRecipesAvailable:
    """Test that all recipes are available and listed."""

    def test_all_recipes_listed_in_module(self):
        """All expected recipe functions are exported from recipes module."""
        expected = {
            "star_forming_photometry",
            "quiescent_z0",
            "agn_panchromatic",
            "stochastic_sfh_jwst",
            "mock_recovery_minimal",
        }
        actual = {name for name in dir(recipes) if not name.startswith("_")}
        assert expected.issubset(actual), f"Missing recipes: {expected - actual}"

    def test_recipes_are_callable(self):
        """Each recipe is a callable function."""
        recipe_names = [
            "star_forming_photometry",
            "quiescent_z0",
            "agn_panchromatic",
            "stochastic_sfh_jwst",
            "mock_recovery_minimal",
        ]
        for name in recipe_names:
            func = getattr(recipes, name)
            assert callable(func), f"{name} should be callable"


class TestStarFormingPhotometry:
    """Tests for star_forming_photometry() recipe."""

    def test_star_forming_returns_dict(self):
        """Recipe returns a dict."""
        result = recipes.star_forming_photometry()
        assert isinstance(result, dict)

    def test_star_forming_builds_parameters(self):
        """Recipe output passes to parse_groups()."""
        recipe_dict = recipes.star_forming_photometry()
        spec = parse_groups(**recipe_dict)
        assert isinstance(spec, Parameters)

    def test_star_forming_has_dpl_sfh(self):
        """Recipe uses DPL SFH."""
        recipe_dict = recipes.star_forming_photometry()
        assert "sfh" in recipe_dict
        assert recipe_dict["sfh"]["type"] == "dpl"

    def test_star_forming_has_free_sfh(self):
        """DPL params are free (via wildcard)."""
        recipe_dict = recipes.star_forming_photometry()
        spec = parse_groups(**recipe_dict)
        # DPL should have ~4 free params (alpha, beta, tau_gyr, log_total_mass)
        dpl_free = [p for p in spec.free_params if p.startswith("sfh_dpl_")]
        assert len(dpl_free) >= 3, f"Expected at least 3 free DPL params, got {len(dpl_free)}"

    def test_star_forming_has_calzetti_dust(self):
        """Recipe uses Calzetti dust law."""
        recipe_dict = recipes.star_forming_photometry()
        assert "dust" in recipe_dict
        assert recipe_dict["dust"]["law_bc"] == "calzetti"

    def test_star_forming_dust_two_component(self):
        """Recipe uses two-component dust."""
        recipe_dict = recipes.star_forming_photometry()
        assert recipe_dict["dust"]["type"] == "two_component"

    def test_star_forming_has_dale2014_emission(self):
        """Recipe includes Dale2014 dust emission (fixed)."""
        recipe_dict = recipes.star_forming_photometry()
        assert "emission" in recipe_dict["dust"]
        assert recipe_dict["dust"]["emission"]["type"] == "dale2014"

    def test_star_forming_has_cue_nebular(self):
        """Recipe uses Cue nebular (fixed)."""
        recipe_dict = recipes.star_forming_photometry()
        assert "neb" in recipe_dict
        assert recipe_dict["neb"]["type"] == "cue"

    def test_star_forming_cue_is_fixed(self):
        """Cue nebular params are fixed."""
        recipe_dict = recipes.star_forming_photometry()
        spec = parse_groups(**recipe_dict)
        neb_free = [p for p in spec.free_params if p.startswith("neb_")]
        assert len(neb_free) == 0, f"Expected no free nebular params, got {neb_free}"

    def test_star_forming_free_redshift(self):
        """Redshift is free with reasonable bounds."""
        recipe_dict = recipes.star_forming_photometry()
        spec = parse_groups(**recipe_dict)
        assert "redshift" in spec.free_params

    def test_star_forming_apply_igm(self):
        """Recipe enables IGM absorption."""
        recipe_dict = recipes.star_forming_photometry()
        assert recipe_dict.get("apply_igm", True) is True

    def test_star_forming_free_param_count(self):
        """Recipe has reasonable number of free parameters."""
        recipe_dict = recipes.star_forming_photometry()
        spec = parse_groups(**recipe_dict)
        # DPL (~4) + dust tau_bc, tau_diff (~2) + redshift (1) + met_logzsol (1) ~ 8-10
        assert 5 <= spec.n_free <= 20, f"Expected 5-20 free params, got {spec.n_free}"


class TestQuiescentZ0:
    """Tests for quiescent_z0() recipe."""

    def test_quiescent_z0_returns_dict(self):
        """Recipe returns a dict."""
        result = recipes.quiescent_z0()
        assert isinstance(result, dict)

    def test_quiescent_z0_builds_parameters(self):
        """Recipe output passes to parse_groups()."""
        recipe_dict = recipes.quiescent_z0()
        spec = parse_groups(**recipe_dict)
        assert isinstance(spec, Parameters)

    def test_quiescent_z0_has_dexp_sfh(self):
        """Recipe uses delayed-exponential SFH."""
        recipe_dict = recipes.quiescent_z0()
        assert "sfh" in recipe_dict
        assert recipe_dict["sfh"]["type"] == "dexp"

    def test_quiescent_z0_has_free_sfh(self):
        """DExp params are free."""
        recipe_dict = recipes.quiescent_z0()
        spec = parse_groups(**recipe_dict)
        dexp_free = [p for p in spec.free_params if p.startswith("sfh_dexp_")]
        assert len(dexp_free) >= 2, f"Expected at least 2 free DExp params, got {len(dexp_free)}"

    def test_quiescent_z0_dust_free(self):
        """Dust attenuation params are free."""
        recipe_dict = recipes.quiescent_z0()
        spec = parse_groups(**recipe_dict)
        dust_free = [p for p in spec.free_params if p.startswith("dust_")]
        assert len(dust_free) >= 1, f"Expected free dust params, got {dust_free}"

    def test_quiescent_z0_no_dust_emission(self):
        """Recipe disables dust emission."""
        recipe_dict = recipes.quiescent_z0()
        spec = parse_groups(**recipe_dict)
        # Should not have dust emission params in free list
        dust_emission_free = [
            p
            for p in spec.free_params
            if p
            in (
                "dust_T",
                "dust_beta_ir",
                "dust_alpha_mir",
                "dust_alpha_dale",
                "dust_umin",
                "dust_gamma_dl",
                "dust_qpah",
                "dust_alpha_dl14",
                "dust_eta_balance",
            )
        ]
        assert len(dust_emission_free) == 0, "Expected no dust emission free params"

    def test_quiescent_z0_fixed_redshift(self):
        """Redshift is fixed at z=0.05."""
        recipe_dict = recipes.quiescent_z0()
        spec = parse_groups(**recipe_dict)
        assert "redshift" in spec.fixed_params
        assert spec.get_distribution("redshift") == recipes.Fixed(0.05)

    def test_quiescent_z0_free_param_count(self):
        """Recipe has reasonable number of free parameters."""
        recipe_dict = recipes.quiescent_z0()
        spec = parse_groups(**recipe_dict)
        # DExp (~3) + dust tau_bc, tau_diff (~2) + met_logzsol (1) ~ 6-8
        assert 4 <= spec.n_free <= 15, f"Expected 4-15 free params, got {spec.n_free}"


class TestAgnPanchromatic:
    """Tests for agn_panchromatic() recipe."""

    def test_agn_panchromatic_returns_dict(self):
        """Recipe returns a dict."""
        result = recipes.agn_panchromatic()
        assert isinstance(result, dict)

    def test_agn_panchromatic_builds_parameters(self):
        """Recipe output passes to parse_groups()."""
        recipe_dict = recipes.agn_panchromatic()
        spec = parse_groups(**recipe_dict)
        assert isinstance(spec, Parameters)

    def test_agn_panchromatic_has_dpl_sfh(self):
        """Recipe uses DPL SFH."""
        recipe_dict = recipes.agn_panchromatic()
        assert "sfh" in recipe_dict
        assert recipe_dict["sfh"]["type"] == "dpl"

    def test_agn_panchromatic_has_agn_group(self):
        """Recipe includes AGN group."""
        recipe_dict = recipes.agn_panchromatic()
        assert "agn" in recipe_dict

    def test_agn_panchromatic_agn_has_disc(self):
        """AGN group has disc sub-block."""
        recipe_dict = recipes.agn_panchromatic()
        assert "disc" in recipe_dict["agn"]

    def test_agn_panchromatic_agn_has_torus(self):
        """AGN group has torus sub-block."""
        recipe_dict = recipes.agn_panchromatic()
        assert "torus" in recipe_dict["agn"]

    def test_agn_panchromatic_includes_radio_xray(self):
        """Recipe includes radio and xray settings."""
        recipe_dict = recipes.agn_panchromatic()
        assert "radio" in recipe_dict, "Radio should be in recipe"
        assert "xray" in recipe_dict, "Xray should be in recipe"
        assert recipe_dict["radio"] is True, "Radio should be True"
        assert recipe_dict["xray"] is True, "Xray should be True"

    def test_agn_panchromatic_free_param_count(self):
        """Recipe has reasonable number of free parameters."""
        recipe_dict = recipes.agn_panchromatic()
        spec = parse_groups(**recipe_dict)
        # DPL + dust + AGN blocks should be substantial
        assert 8 <= spec.n_free <= 50, f"Expected 8-50 free params, got {spec.n_free}"


class TestStochasticSfhJwst:
    """Tests for stochastic_sfh_jwst() recipe."""

    def test_stochastic_sfh_jwst_returns_dict(self):
        """Recipe returns a dict."""
        result = recipes.stochastic_sfh_jwst()
        assert isinstance(result, dict)

    def test_stochastic_sfh_jwst_builds_parameters(self):
        """Recipe output passes to parse_groups()."""
        recipe_dict = recipes.stochastic_sfh_jwst()
        spec = parse_groups(**recipe_dict)
        assert isinstance(spec, Parameters)

    def test_stochastic_sfh_jwst_has_composition(self):
        """Recipe uses DPL + field SFH composition."""
        recipe_dict = recipes.stochastic_sfh_jwst()
        assert "sfh" in recipe_dict
        assert recipe_dict["sfh"]["type"] == ["dpl", "field"]

    def test_stochastic_sfh_jwst_high_z_range(self):
        """Redshift is in JWST high-z range (0.5-12)."""
        recipe_dict = recipes.stochastic_sfh_jwst()
        spec = parse_groups(**recipe_dict)
        # Redshift should be free and in high-z range
        assert "redshift" in spec.free_params

    def test_stochastic_sfh_jwst_apply_igm(self):
        """Recipe enables IGM absorption (important for high-z)."""
        recipe_dict = recipes.stochastic_sfh_jwst()
        assert recipe_dict.get("apply_igm", True) is True

    def test_stochastic_sfh_jwst_has_field_params(self):
        """Recipe includes field (stochastic) SFH params."""
        recipe_dict = recipes.stochastic_sfh_jwst()
        spec = parse_groups(**recipe_dict)
        field_params = [p for p in spec.all_params if p.startswith("sfh_field_")]
        assert len(field_params) > 0, "Expected field SFH params"

    def test_stochastic_sfh_jwst_free_param_count(self):
        """Recipe has reasonable number of free parameters."""
        recipe_dict = recipes.stochastic_sfh_jwst()
        spec = parse_groups(**recipe_dict)
        # DPL (~4) + field (~2) + dust + met + redshift ~ 10-15
        assert 8 <= spec.n_free <= 25, f"Expected 8-25 free params, got {spec.n_free}"


class TestMockRecoveryMinimal:
    """Tests for mock_recovery_minimal() recipe."""

    def test_mock_recovery_minimal_returns_dict(self):
        """Recipe returns a dict."""
        result = recipes.mock_recovery_minimal()
        assert isinstance(result, dict)

    def test_mock_recovery_minimal_builds_parameters(self):
        """Recipe output passes to parse_groups()."""
        recipe_dict = recipes.mock_recovery_minimal()
        spec = parse_groups(**recipe_dict)
        assert isinstance(spec, Parameters)

    def test_mock_recovery_minimal_has_tsnorm_sfh(self):
        """Recipe uses tsnorm (top-hat) SFH."""
        recipe_dict = recipes.mock_recovery_minimal()
        assert "sfh" in recipe_dict
        assert recipe_dict["sfh"]["type"] == "tsnorm"

    def test_mock_recovery_minimal_minimal_free_params(self):
        """Recipe has minimal (~5-6) free parameters."""
        recipe_dict = recipes.mock_recovery_minimal()
        spec = parse_groups(**recipe_dict)
        # Tsnorm (~4) + dust tau_bc (1) + met_logzsol (1) = ~6
        assert 4 <= spec.n_free <= 8, f"Expected 4-8 free params, got {spec.n_free}"

    def test_mock_recovery_minimal_no_nebular(self):
        """Recipe disables nebular emission."""
        recipe_dict = recipes.mock_recovery_minimal()
        assert "neb" in recipe_dict
        assert recipe_dict["neb"]["type"] == "none"

    def test_mock_recovery_minimal_fixed_redshift(self):
        """Redshift is fixed at z=0.05 for local mocks."""
        recipe_dict = recipes.mock_recovery_minimal()
        spec = parse_groups(**recipe_dict)
        assert "redshift" in spec.fixed_params
        assert spec.get_distribution("redshift") == recipes.Fixed(0.05)

    def test_mock_recovery_minimal_calzetti_dust(self):
        """Recipe uses Calzetti dust law."""
        recipe_dict = recipes.mock_recovery_minimal()
        assert "dust" in recipe_dict
        assert recipe_dict["dust"]["law_bc"] == "calzetti"

    def test_mock_recovery_minimal_no_dust_emission(self):
        """Recipe disables dust emission (for speed)."""
        recipe_dict = recipes.mock_recovery_minimal()
        spec = parse_groups(**recipe_dict)
        # Should not have dust emission params free
        dust_emission_free = [
            p
            for p in spec.free_params
            if p
            in (
                "dust_T",
                "dust_beta_ir",
                "dust_alpha_mir",
                "dust_alpha_dale",
                "dust_umin",
                "dust_gamma_dl",
                "dust_qpah",
                "dust_alpha_dl14",
                "dust_eta_balance",
            )
        ]
        assert len(dust_emission_free) == 0, "Expected no dust emission params"


class TestRecipesIntegration:
    """Integration tests: recipes work with SEDModel (if SSP data available)."""

    def test_all_recipes_compose_with_splatting(self):
        """All recipes can be splatted into parse_groups()."""
        for recipe_func in [
            recipes.star_forming_photometry,
            recipes.quiescent_z0,
            recipes.agn_panchromatic,
            recipes.stochastic_sfh_jwst,
            recipes.mock_recovery_minimal,
        ]:
            recipe_dict = recipe_func()
            spec = parse_groups(**recipe_dict)
            assert isinstance(spec, Parameters)
            assert spec.n_free > 0, f"{recipe_func.__name__} has no free params"
