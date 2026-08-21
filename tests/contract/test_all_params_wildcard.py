# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Contract tests for the all_params wildcard naming consistency.

Validates that:
- Builders accept all_params= as the canonical spelling
- defaults= on builders still works and warns (deprecated)
- Dict grammar rejects defaults= with guidance to use all_params=
- Builder and dict grammar spellings produce identical models
"""

import pytest

from tengri import (
    FIXED,
    FREE,
    Fixed,
    SEDModel,
    Uniform,
    builders,
)
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


# Parametrize over multiple builder types to ensure all accept all_params=
BUILDER_VARIANTS = [
    pytest.param(
        lambda: builders.sfh.dpl(all_params=FREE),
        "sfh.dpl",
        id="sfh_dpl",
    ),
    pytest.param(
        lambda: builders.dust.two_component(all_params=FIXED, law="calzetti"),
        "dust.two_component",
        id="dust_two_component",
    ),
    pytest.param(
        lambda: builders.neb.cue(all_params=FIXED),
        "neb.cue",
        id="neb_cue",
    ),
]


class TestBuilderAllParams:
    """Test that builders accept all_params= as the canonical spelling."""

    def test_builder_all_params_canonical(self):
        """Builder accepts all_params= and returns correct dict."""
        sfh_dict = builders.sfh.dpl(all_params=FREE)
        assert sfh_dict["type"] == "dpl"
        assert sfh_dict["all_params"] is FREE

    def test_builder_all_params_with_override(self):
        """Builder with all_params= and per-param overrides."""
        sfh_dict = builders.sfh.dpl(all_params=FIXED, alpha=Uniform(0.5, 3.0))
        assert sfh_dict["type"] == "dpl"
        assert sfh_dict["all_params"] is FIXED
        assert sfh_dict["alpha"] == Uniform(0.5, 3.0)

    def test_builder_defaults_deprecated_alias(self):
        """Builder accepts deprecated defaults= with a warning."""
        with pytest.warns(DeprecationWarning, match=r"defaults=.*all_params="):
            sfh_dict = builders.sfh.dpl(defaults=FREE)
        assert sfh_dict["type"] == "dpl"
        assert sfh_dict["all_params"] is FREE

    def test_builder_rejects_both_all_params_and_defaults(self):
        """Builder raises if both all_params= and defaults= are passed."""
        with pytest.raises(TypeError, match=r"all_params=.*defaults=.*not both"):
            builders.sfh.dpl(all_params=FREE, defaults=FIXED)

    def test_builder_legacy_underscore_still_works(self):
        """Builder accepts legacy _= with a deprecation warning."""
        with pytest.warns(DeprecationWarning, match=r"_=.*all_params="):
            sfh_dict = builders.sfh.dpl(_=FREE)
        assert sfh_dict["type"] == "dpl"
        assert sfh_dict["all_params"] is FREE

    def test_builder_rejects_all_params_and_underscore(self):
        """Builder raises if both all_params= and _= are passed."""
        with pytest.raises(TypeError, match=r"all_params=.*_=.*not both"):
            builders.sfh.dpl(all_params=FREE, _=FIXED)

    def test_builder_rejects_defaults_and_underscore(self):
        """Builder raises if both defaults= and _= are passed."""
        # The TypeError is raised, and we don't emit a warning because
        # the error happens before the deprecation path
        with pytest.raises(TypeError, match=r"defaults=.*_=.*not both"):
            builders.sfh.dpl(defaults=FREE, _=FIXED)


class TestDictGrammarAllParams:
    """Test that dict grammar validates all_params correctly."""

    def test_dict_grammar_accepts_all_params(self):
        """Dict grammar accepts all_params as a structural key."""
        params = parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": FREE},
        )
        assert "sfh_dpl_alpha" in params.free_params

    def test_dict_grammar_rejects_defaults_with_guidance(self):
        """Dict grammar rejects 'defaults' and names 'all_params' as correct."""
        with pytest.raises(ValueError, match=r"'defaults'.*all_params"):
            parse_groups(
                redshift=Fixed(0.1),
                sfh={"type": "dpl", "defaults": FREE},
            )

    def test_dict_grammar_accepts_wildcard_alias(self):
        """Dict grammar still accepts '*' wildcard alias."""
        params = parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "*": FREE},
        )
        assert "sfh_dpl_alpha" in params.free_params


class TestMultipleBuilderTypes:
    """Test that all builder types accept all_params= and deprecate defaults=."""

    @pytest.mark.parametrize("builder_fn,name", BUILDER_VARIANTS)
    def test_all_builders_accept_all_params(self, builder_fn, name):
        """All builder types accept all_params= as canonical."""
        result = builder_fn()
        assert result["all_params"] in (FREE, FIXED), f"{name} should have all_params key"

    def test_dust_all_params_canonical(self):
        """dust.two_component accepts all_params= canonically."""
        dust_dict = builders.dust.two_component(all_params=FIXED, law="calzetti")
        assert dust_dict["type"] == "two_component"
        assert dust_dict["all_params"] is FIXED

    def test_dust_defaults_deprecated(self):
        """dust.two_component accepts deprecated defaults= with warning."""
        with pytest.warns(DeprecationWarning, match=r"defaults=.*all_params="):
            dust_dict = builders.dust.two_component(defaults=FIXED, law="calzetti")
        assert dust_dict["type"] == "two_component"
        assert dust_dict["all_params"] is FIXED

    def test_neb_all_params_canonical(self):
        """neb.cue accepts all_params= canonically."""
        neb_dict = builders.neb.cue(all_params=FIXED)
        assert neb_dict["type"] == "cue"
        assert neb_dict["all_params"] is FIXED

    def test_neb_defaults_deprecated(self):
        """neb.cue accepts deprecated defaults= with warning."""
        with pytest.warns(DeprecationWarning, match=r"defaults=.*all_params="):
            neb_dict = builders.neb.cue(defaults=FIXED)
        assert neb_dict["type"] == "cue"
        assert neb_dict["all_params"] is FIXED


class TestBuilderDictEquivalence:
    """Test that builder and dict spellings produce identical models."""

    def test_builder_and_dict_free_params_match(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Models built via builder and dict have identical free_params."""
        # Builder spelling
        model_builder = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.1),
            sfh=builders.sfh.dpl(all_params=FREE),
        )

        # Dict spelling
        model_dict = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": FREE},
        )

        assert model_builder.spec.free_params == model_dict.spec.free_params

    def test_builder_and_dict_fixed_params_match(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Models built via builder and dict have identical free_params (FIXED case)."""
        # Builder spelling
        model_builder = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.1),
            sfh=builders.sfh.dpl(all_params=FIXED),
        )

        # Dict spelling
        model_dict = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": FIXED},
        )

        assert model_builder.spec.free_params == model_dict.spec.free_params

    def test_builder_and_dict_mixed_policy_match(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Builder and dict produce identical specs with mixed free/fixed."""
        # Builder spelling
        model_builder = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.1),
            sfh=builders.sfh.dpl(all_params=FREE, alpha=Fixed(1.5)),
        )

        # Dict spelling
        model_dict = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": FREE, "alpha": Fixed(1.5)},
        )

        assert model_builder.spec.free_params == model_dict.spec.free_params

    def test_deprecated_defaults_produces_same_model(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Deprecated defaults= produces the same model as all_params=."""
        with pytest.warns(DeprecationWarning):
            model_deprecated = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                redshift=Fixed(0.1),
                sfh=builders.sfh.dpl(defaults=FREE),
            )

        model_canonical = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.1),
            sfh=builders.sfh.dpl(all_params=FREE),
        )

        assert model_deprecated.spec.free_params == model_canonical.spec.free_params
