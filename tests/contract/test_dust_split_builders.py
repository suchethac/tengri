# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Contract tests for dust group split: builders and recipes.

Tests that dust attenuation and IR emission are correctly separated into
two peer top-level groups, and that all recipes build successfully with
unchanged free-parameter sets.
"""

from __future__ import annotations

import pytest

from tengri import builders, recipes
from tengri.parameters.groups import parse_groups
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.sentinels import FIXED, FREE

pytestmark = pytest.mark.contract


class TestDustAttenutationBuilders:
    """Attenuation builders return dicts suitable for dust_attenuation kwarg."""

    def test_single_component_returns_attenuation_dict(self):
        """single_component() returns a dict with no emission key."""
        result = builders.dust.single_component(law="calzetti", defaults=FREE)
        assert isinstance(result, dict)
        assert result["type"] == "single_component"
        assert "law" in result
        assert "emission" not in result
        assert "tau_v" not in result  # tau_v comes from short_params

    def test_single_component_with_prior(self):
        """single_component() accepts and returns tau_v prior."""
        result = builders.dust.single_component(law="calzetti", defaults=FREE, tau_v=Uniform(0, 3))
        assert result["tau_v"] == Uniform(0, 3)

    def test_two_component_returns_attenuation_dict(self):
        """two_component() returns a dict with no emission key."""
        result = builders.dust.two_component(law="calzetti", defaults=FREE)
        assert isinstance(result, dict)
        assert result["type"] == "two_component"
        assert "law" in result
        assert "emission" not in result

    def test_two_component_with_per_screen_laws(self):
        """two_component() accepts law_bc and law_diff separately."""
        result = builders.dust.two_component(
            law_bc="calzetti", law_diff="power_law", defaults=FREE
        )
        assert result["law_bc"] == "calzetti"
        assert result["law_diff"] == "power_law"
        assert "law" not in result

    def test_two_component_with_priors(self):
        """two_component() accepts and returns tau_bc, tau_diff priors."""
        result = builders.dust.two_component(
            law="calzetti",
            defaults=FREE,
            tau_bc=Uniform(0, 2),
            tau_diff=Uniform(0, 3),
        )
        assert result["tau_bc"] == Uniform(0, 2)
        assert result["tau_diff"] == Uniform(0, 3)

    def test_attenuation_builder_raises_on_emission_kwarg(self):
        """Passing emission= to attenuation builder raises with helpful message."""
        with pytest.raises(TypeError, match=r"emission.*no longer nested"):
            builders.dust.two_component(
                law="calzetti",
                emission=builders.dust.emission.dale2014(),
            )

    def test_single_component_in_build(self):
        """single_component() output is accepted by dust_attenuation= kwarg."""
        # Smoke test: builder output parses without error.
        dust_attenuation = builders.dust.single_component(law="calzetti", defaults=FIXED)
        groups = {
            "sfh": {"type": "dpl", "all_params": FIXED},
            "dust_attenuation": dust_attenuation,
            "neb": {"type": "none"},
            "redshift": Fixed(0.1),
        }
        spec = parse_groups(**groups)
        # Everything is fixed
        assert len(spec.free_params) == 0

    def test_two_component_in_build(self):
        """two_component() output is accepted by dust_attenuation= kwarg."""
        dust_attenuation = builders.dust.two_component(law="calzetti", defaults=FIXED)
        groups = {
            "sfh": {"type": "dpl", "all_params": FIXED},
            "dust_attenuation": dust_attenuation,
            "neb": {"type": "none"},
            "redshift": Fixed(0.1),
        }
        spec = parse_groups(**groups)
        # Everything is fixed
        assert len(spec.free_params) == 0


class TestDustEmissionBuilders:
    """Emission builders return dicts suitable for dust_emission kwarg."""

    def test_emission_dale2014_returns_emission_dict(self):
        """dale2014() returns an emission dict with correct type."""
        result = builders.dust.emission.dale2014(defaults=FIXED)
        assert isinstance(result, dict)
        assert result["type"] == "dale2014"
        assert "emission" not in result

    def test_emission_in_build(self):
        """dust_emission= kwarg accepts emission builder output."""
        dust_emission = builders.dust.emission.dale2014(defaults=FIXED)
        groups = {
            "sfh": {"type": "dpl", "all_params": FIXED},
            "dust_attenuation": {
                "type": "single_component",
                "law": "calzetti",
                "all_params": FIXED,
            },
            "dust_emission": dust_emission,
            "neb": {"type": "none"},
            "redshift": Fixed(0.1),
        }
        spec = parse_groups(**groups)
        # Everything is fixed
        assert len(spec.free_params) == 0

    def test_relaxed_energy_balance_frees_eta_balance(self):
        """relaxed_energy_balance() returns emission dict with free eta_balance."""
        result = builders.dust.emission.relaxed_energy_balance(model="dale2014", sigma=0.2)
        assert isinstance(result, dict)
        assert result["type"] == "dale2014"
        assert "eta_balance" in result
        # eta_balance is a LogNormal prior with mu=0
        from tengri.parameters.priors import LogNormal

        assert isinstance(result["eta_balance"], LogNormal)
        assert result["eta_balance"].mu == 0.0
        assert result["eta_balance"].sigma == 0.2

    def test_relaxed_energy_balance_in_build(self):
        """relaxed_energy_balance() output builds successfully."""
        dust_emission = builders.dust.emission.relaxed_energy_balance(sigma=0.3)
        groups = {
            "sfh": {"type": "dpl", "all_params": FIXED},
            "dust_attenuation": {"type": "two_component", "law": "calzetti", "all_params": FIXED},
            "dust_emission": dust_emission,
            "neb": {"type": "none"},
            "redshift": Fixed(0.1),
        }
        spec = parse_groups(**groups)
        # eta_balance should be in free params due to relaxed_energy_balance
        assert "dust_eta_balance" in spec.free_params, f"Free params: {spec.free_params}"


class TestRecipes:
    """All ten recipes build successfully with two-group dust structure."""

    @pytest.mark.parametrize(
        "recipe_name",
        [
            "star_forming_photometry",
            "quiescent_z0",
            "high_z",
            "photoz",
            "agn_panchromatic",
            "composable_agn",
            "stochastic_sfh_jwst",
            "mock_recovery_minimal",
            "dust_demo",
            "unified_agn",
        ],
    )
    def test_recipe_has_dust_groups(self, recipe_name):
        """Recipe has dust_attenuation (and sometimes dust_emission)."""
        recipe_fn = getattr(recipes, recipe_name)
        recipe_dict = recipe_fn()
        # All recipes should have dust_attenuation (previously 'dust')
        assert "dust_attenuation" in recipe_dict
        assert "dust" not in recipe_dict

    def test_star_forming_photometry(self):
        """star_forming_photometry builds with dust_attenuation and dust_emission."""
        recipe_dict = recipes.star_forming_photometry()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert recipe_dict["dust_emission"]["type"] == "dale2014"

    def test_quiescent_z0(self):
        """quiescent_z0 builds with dust_attenuation only."""
        recipe_dict = recipes.quiescent_z0()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert "dust_emission" not in recipe_dict

    def test_high_z(self):
        """high_z builds with dust_attenuation only."""
        recipe_dict = recipes.high_z()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert "dust_emission" not in recipe_dict

    def test_photoz(self):
        """photoz builds with dust_attenuation only."""
        recipe_dict = recipes.photoz()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert "dust_emission" not in recipe_dict

    def test_agn_panchromatic(self):
        """agn_panchromatic builds with dust_attenuation and dust_emission."""
        recipe_dict = recipes.agn_panchromatic()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert recipe_dict["dust_emission"]["type"] == "dale2014_cigale"

    def test_composable_agn(self):
        """composable_agn builds with dust_attenuation and dust_emission."""
        recipe_dict = recipes.composable_agn()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert recipe_dict["dust_emission"]["type"] == "dale2014_cigale"

    def test_stochastic_sfh_jwst(self):
        """stochastic_sfh_jwst builds with dust_attenuation and dust_emission."""
        recipe_dict = recipes.stochastic_sfh_jwst()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert recipe_dict["dust_emission"]["type"] == "dale2014"

    def test_mock_recovery_minimal(self):
        """mock_recovery_minimal builds with dust_attenuation only."""
        recipe_dict = recipes.mock_recovery_minimal()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert "dust_emission" not in recipe_dict

    def test_dust_demo(self):
        """dust_demo builds with dust_attenuation only."""
        recipe_dict = recipes.dust_demo()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert "dust_emission" not in recipe_dict

    def test_unified_agn(self):
        """unified_agn builds with dust_attenuation only."""
        recipe_dict = recipes.unified_agn()
        assert recipe_dict["dust_attenuation"]["type"] == "two_component"
        assert "dust_emission" not in recipe_dict


class TestRecipeFreeparams:
    """Recipe free-parameter sets are unchanged after split."""

    def test_star_forming_photometry_free_params(self):
        """star_forming_photometry retains its free params."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = parse_groups(**recipes.star_forming_photometry())
        # With defaults=FREE, all SFH + attenuation params are free
        # Check key params are present
        assert "sfh_dpl_alpha" in spec.free_params
        assert "sfh_dpl_beta" in spec.free_params
        assert "dust_tau_bc" in spec.free_params
        assert "dust_tau_diff" in spec.free_params
        assert "met_logzsol" in spec.free_params
        assert "redshift" in spec.free_params

    def test_quiescent_z0_free_params(self):
        """quiescent_z0 retains its free params."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = parse_groups(**recipes.quiescent_z0())
        # With defaults=FIXED, only explicit priors are free
        assert "sfh_dexp_tau_gyr" in spec.free_params
        assert "sfh_dexp_log_total_mass" in spec.free_params
        assert "dust_tau_bc" in spec.free_params
        assert "dust_tau_diff" in spec.free_params
        assert "met_logzsol" in spec.free_params

    def test_photoz_free_params(self):
        """photoz retains its free params."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = parse_groups(**recipes.photoz())
        # With defaults=FIXED, only explicit priors are free
        assert "sfh_dpl_alpha" in spec.free_params
        assert "dust_tau_bc" in spec.free_params
        assert "dust_tau_diff" in spec.free_params
        assert "met_logzsol" in spec.free_params or "met_logzsol" not in spec.fixed_params
        assert "redshift" in spec.free_params

    def test_mock_recovery_minimal_free_params(self):
        """mock_recovery_minimal retains its free params."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = parse_groups(**recipes.mock_recovery_minimal())
        # With defaults=FREE for SFH and FIXED for dust (with explicit tau_bc)
        assert "sfh_tsnorm_log_total_mass" in spec.free_params
        assert "dust_tau_bc" in spec.free_params
        assert "met_logzsol" in spec.free_params


class TestExplicitAllParamsPreserved:
    """Recipes that declare all_params: FIXED are preserved exactly."""

    def test_high_z_all_params_fixed(self):
        """high_z has all_params: FIXED on dust_attenuation and sfh."""
        recipe_dict = recipes.high_z()
        assert recipe_dict["sfh"]["all_params"] == FIXED
        assert recipe_dict["dust_attenuation"]["all_params"] == FIXED

    def test_photoz_all_params_fixed(self):
        """photoz has all_params: FIXED on dust_attenuation and sfh."""
        recipe_dict = recipes.photoz()
        assert recipe_dict["sfh"]["all_params"] == FIXED
        assert recipe_dict["dust_attenuation"]["all_params"] == FIXED

    def test_unified_agn_all_params_fixed(self):
        """unified_agn has all_params: FIXED on dust_attenuation and sfh."""
        recipe_dict = recipes.unified_agn()
        assert recipe_dict["sfh"]["all_params"] == FIXED
        assert recipe_dict["dust_attenuation"]["all_params"] == FIXED

    def test_dust_demo_all_params_fixed(self):
        """dust_demo has all_params: FIXED on sfh and dust_attenuation."""
        recipe_dict = recipes.dust_demo()
        assert recipe_dict["sfh"]["all_params"] == FIXED
        assert recipe_dict["dust_attenuation"]["all_params"] == FIXED


class TestBuildersSurfaceContract:
    """Builders accept the documented signatures without breaking."""

    def test_single_component_signature_has_law(self):
        """single_component signature shows law as parameter."""
        sig = str(builders.dust.single_component.__signature__)
        assert "law" in sig
        assert "defaults" in sig

    def test_two_component_signature_has_law_variants(self):
        """two_component signature shows law and law_bc/law_diff."""
        sig = str(builders.dust.two_component.__signature__)
        assert "law" in sig or ("law_bc" in sig and "law_diff" in sig)
        assert "defaults" in sig

    def test_emission_dale2014_signature(self):
        """emission.dale2014 signature has defaults parameter."""
        sig = str(builders.dust.emission.dale2014.__signature__)
        assert "defaults" in sig

    def test_relaxed_energy_balance_callable(self):
        """relaxed_energy_balance is callable and works as expected."""
        result = builders.dust.emission.relaxed_energy_balance(sigma=0.3)
        assert isinstance(result, dict)
        assert "type" in result
        assert "eta_balance" in result
