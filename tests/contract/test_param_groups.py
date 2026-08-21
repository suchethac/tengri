# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parse_groups() nested-dict model builder.

This module tests the translation from Bagpipes-style nested dicts
to Parameters objects via parse_groups().
"""

import pytest

from tengri.parameters import FIXED, FREE, Fixed, Uniform
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract


class TestWildcardValidation:
    """The '*' wildcard slot must be FREE or FIXED — strings/None/bools error."""

    def test_string_free_raises(self):
        with pytest.raises(ValueError, match="all_params"):
            parse_groups(sfh={"type": "dpl", "*": "free"}, redshift=Fixed(0.1))

    def test_string_fixed_raises(self):
        with pytest.raises(ValueError, match="all_params"):
            parse_groups(sfh={"type": "dpl", "*": "fixed"}, redshift=Fixed(0.1))

    def test_none_raises(self):
        with pytest.raises(ValueError, match="all_params"):
            parse_groups(sfh={"type": "dpl", "*": None}, redshift=Fixed(0.1))

    def test_bool_raises(self):
        with pytest.raises(ValueError, match="all_params"):
            parse_groups(sfh={"type": "dpl", "*": True}, redshift=Fixed(0.1))


class TestWildcard:
    """Test wildcard ('*') semantics for per-group parameter selection."""

    def test_star_free_frees_all_declared_params(self):
        """With '*': FREE, all params in the group should be free."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FREE},
            redshift=Fixed(0.1),
        )
        assert isinstance(params, Parameters)
        # dpl has: alpha, beta, tau_gyr, log_total_mass
        assert "sfh_dpl_alpha" in params.free_params
        assert "sfh_dpl_beta" in params.free_params
        assert "sfh_dpl_tau_gyr" in params.free_params
        assert "sfh_dpl_log_total_mass" in params.free_params

    def test_star_fixed_fixes_all_declared_params(self):
        """With '*': FIXED, all params in the group should be fixed."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert isinstance(params, Parameters)
        # dpl params should be in fixed_params because they'll be fixed at registry defaults
        assert "sfh_dpl_alpha" in params.fixed_params
        assert "sfh_dpl_beta" in params.fixed_params
        assert "sfh_dpl_tau_gyr" in params.fixed_params
        assert "sfh_dpl_log_total_mass" in params.fixed_params

    def test_star_omitted_defaults_to_fixed(self):
        """When '*' is not present, default behavior is to fix all params."""
        params = parse_groups(
            sfh={"type": "dpl"},
            redshift=Fixed(0.1),
        )
        # All dpl params should be fixed at registry defaults
        assert "sfh_dpl_alpha" in params.fixed_params
        assert "sfh_dpl_beta" in params.fixed_params

    def test_per_param_override_beats_wildcard(self):
        """Per-parameter override should win over the wildcard."""
        params = parse_groups(
            sfh={
                "type": "dpl",
                "*": FREE,
                "beta": Fixed(2.0),
            },
            redshift=Fixed(0.1),
        )
        # beta should be fixed, others free
        assert "sfh_dpl_beta" in params.fixed_params
        assert params.get_distribution("sfh_dpl_beta").value == 2.0
        assert "sfh_dpl_alpha" in params.free_params
        assert "sfh_dpl_tau_gyr" in params.free_params

    def test_bare_value_becomes_fixed(self):
        """Bare value (e.g., 2.0) should be converted to Fixed(2.0)."""
        params = parse_groups(
            sfh={
                "type": "dpl",
                "beta": 2.0,
            },
            redshift=Fixed(0.1),
        )
        dist = params.get_distribution("sfh_dpl_beta")
        assert isinstance(dist, Fixed)
        assert dist.value == 2.0


class TestEquivalence:
    """Test that grouped form is equivalent to flat form."""

    def test_grouped_equals_flat_dpl_minimal(self):
        """Minimal grouped dpl should equal equivalent flat form (#1796).

        When there's no met block, met_* parameters are implicitly FIXED.
        """
        # When dust group is NOT specified, dust defaults to free params
        grouped = parse_groups(
            sfh={
                "type": "dpl",
                "alpha": Uniform(0.5, 3.0),
                "beta": Fixed(1.0),
                "*": FREE,
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "*": FIXED,
            },
            redshift=Fixed(0.1),
        )

        flat = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Fixed(1.0),
            # tau_gyr and log_total_mass should be free via registry defaults
            dust_model="two_component",
            dust_tau_bc=0.0,  # Registry default
            dust_tau_diff=0.0,  # Registry default
            # met_logzsol is Fixed when no met block is specified (#1796)
            met_logzsol=Fixed(0.0),
            redshift=Fixed(0.1),
        )

        assert grouped.free_params == flat.free_params
        assert grouped.fixed_params == flat.fixed_params

    def test_grouped_equals_flat_full_panchromatic(self):
        """Full panchromatic grouped model should match flat equivalent."""
        grouped = parse_groups(
            sfh={
                "type": "dpl",
                "*": FREE,
                "beta": Fixed(1.5),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            dust_emission={
                "type": "dale2014",
                "*": FIXED,
            },
            neb={"type": "cue", "*": FIXED},
            igm={"type": "madau"},
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
            xray={"type": "simple"},
            redshift=Uniform(0.01, 10.0),
        )

        # Verify the grouped form has correct settings
        assert grouped.mean_sfh_type == ["dpl"]
        assert grouped.dust_model == "two_component"
        assert grouped.dust_law_bc == "calzetti"
        assert grouped.dust_emission == "dale2014"
        assert grouped.nebular_mode == "cue"
        assert grouped.apply_igm is True
        assert grouped.radio is True
        assert grouped.xray is True

        # Verify param distributions
        assert "sfh_dpl_alpha" in grouped.free_params
        assert "sfh_dpl_beta" in grouped.fixed_params
        assert grouped.get_distribution("sfh_dpl_beta").value == 1.5
        assert grouped.get_distribution("dust_tau_bc").value == 0.5
        assert "redshift" in grouped.free_params


class TestNesting:
    """Test nested sub-blocks (dust.emission, etc.)."""

    def test_dust_emission_subblock(self):
        """dust.emission nested sub-block should activate dust IR params."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "*": FIXED,
                # FIXED, not FREE: every Dale+2014 param has a Fixed registry
                # default, so FREE here frees nothing and is now refused. This
                # test is about sub-block *declaration*, not freeing.
            },
            dust_emission={
                "type": "dale2014",
                "*": FIXED,
                "alpha_dale": Uniform(0.5, 4.0),
            },
            redshift=Fixed(0.1),
        )
        # dust_emission params should be in the registry
        assert "dust_alpha_dale" in params.all_params
        assert "dust_umin" in params.all_params
        # The alpha_dale is free since we specified it explicitly
        assert "dust_alpha_dale" in params.free_params

    def test_dust_emission_omitted_means_no_ir(self):
        """Absence of dust.emission key should not activate IR params."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "*": FIXED,
            },
            redshift=Fixed(0.1),
        )
        # dust_emission params should not be in the registry
        assert "dust_alpha_dale" not in params.all_params
        assert "dust_umin" not in params.all_params


class TestTypeMapping:
    """Test type-to-settings mapping for each group."""

    def test_neb_cue(self):
        """neb={'type': 'cue'} should set nebular_cue=True."""
        params = parse_groups(
            neb={"type": "cue", "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert params.nebular_mode == "cue"
        assert "neb_logU" in params.all_params

    def test_neb_ssp(self):
        """neb={'type': 'ssp'} should set nebular_ssp=True."""
        params = parse_groups(
            neb={"type": "ssp", "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert params.nebular_mode == "ssp"

    def test_neb_none(self):
        """neb={'type': 'none'} or absent neb should disable nebular."""
        params1 = parse_groups(
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
        assert params1.nebular_mode == "off"

        params2 = parse_groups(redshift=Fixed(0.1))
        assert params2.nebular_mode == "off"

    def test_neb_cb19(self):
        """neb={'type': 'cb19'} should set nebular='cb19'."""
        params = parse_groups(
            neb={"type": "cb19", "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert params.nebular_mode == "cb19"
        assert "neb_log_nH" in params.all_params

    def test_igm_madau_sets_apply_igm(self):
        """igm={'type': 'madau'} should set apply_igm=True."""
        params = parse_groups(
            igm={"type": "madau"},
            redshift=Fixed(0.1),
        )
        assert params.apply_igm is True

    def test_igm_inoue14_sets_apply_igm(self):
        """igm={'type': 'inoue14'} should set apply_igm=True."""
        params = parse_groups(
            igm={"type": "inoue14"},
            redshift=Fixed(0.1),
        )
        assert params.apply_igm is True

    def test_igm_none_disables_apply_igm(self):
        """igm={'type': 'none'} should set apply_igm=False."""
        params = parse_groups(
            igm={"type": "none"},
            redshift=Fixed(0.1),
        )
        assert params.apply_igm is False

    def test_igm_patchy_option(self):
        """igm={'type': 'madau', 'patchy': True} should enable patchy IGM params."""
        params = parse_groups(
            igm={"type": "madau", "patchy": True},
            redshift=Fixed(0.1),
        )
        assert params.igm_patchy is True
        assert "igm_x_HI" in params.all_params

    def test_igm_dla_option(self):
        """igm={'type': 'madau', 'dla': True} should enable DLA params."""
        params = parse_groups(
            igm={"type": "madau", "dla": True},
            redshift=Fixed(0.1),
        )
        assert params.dla is True
        assert "dla_log_n_hi" in params.all_params

    def test_igm_model_madau_propagated(self):
        """Regression for #344: igm={'type': 'madau'} must set igm_model='madau'."""
        params = parse_groups(igm={"type": "madau"}, redshift=Fixed(0.1))
        assert params.igm_model == "madau"

    def test_igm_model_inoue14_propagated(self):
        """Regression for #344: igm={'type': 'inoue14'} must select Inoue, not the default."""
        params = parse_groups(igm={"type": "inoue14"}, redshift=Fixed(0.1))
        assert params.igm_model == "inoue"

    def test_igm_model_inoue_alias_propagated(self):
        """The short alias 'inoue' resolves to the canonical 'inoue' name."""
        params = parse_groups(igm={"type": "inoue"}, redshift=Fixed(0.1))
        assert params.igm_model == "inoue"

    def test_igm_none_does_not_set_model(self):
        """igm={'type': 'none'} must not error and must leave igm_model at its default."""
        params = parse_groups(igm={"type": "none"}, redshift=Fixed(0.1))
        assert params.apply_igm is False

    def test_radio_condon92(self):
        """Composable radio form should set radio=True."""
        params = parse_groups(
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
            redshift=Fixed(0.1),
        )
        assert params.radio is True
        assert "radio_q_ir" in params.all_params

    def test_radio_none_or_absent(self):
        """Composable radio with both 'none' or absent should set radio=False."""
        params1 = parse_groups(
            radio={"sf": {"type": "none"}, "agn": {"type": "none"}},
            redshift=Fixed(0.1),
        )
        assert params1.radio is False

        params2 = parse_groups(redshift=Fixed(0.1))
        assert params2.radio is False

    def test_xray_simple(self):
        """xray={'type': 'simple'} should set xray=True."""
        params = parse_groups(
            xray={"type": "simple"},
            redshift=Fixed(0.1),
        )
        assert params.xray is True
        assert "xray_gamma_agn" in params.all_params

    def test_xray_none_or_absent(self):
        """xray={'type': 'none'} or absent xray should set xray=False."""
        params1 = parse_groups(
            xray={"type": "none"},
            redshift=Fixed(0.1),
        )
        assert params1.xray is False

        params2 = parse_groups(redshift=Fixed(0.1))
        assert params2.xray is False

    def test_dust_law_mapping(self):
        """dust={'type': ..., 'law_bc': ...} should set dust_law_bc."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
            },
            redshift=Fixed(0.1),
        )
        assert params.dust_law_bc == "calzetti"

    def test_dust_single_component(self):
        """dust={'type': 'single_component'} should set dust_model."""
        params = parse_groups(
            dust_attenuation={
                "law": "power_law",
                "type": "single_component",
                "*": FIXED,
            },
            redshift=Fixed(0.1),
        )
        assert params.dust_model == "single_component"
        assert "dust_tau_v" in params.all_params


class TestValidation:
    """Test error handling for invalid inputs."""

    def test_unknown_group_key_raises_value_error(self):
        """Unknown group key should raise ValueError with suggestions."""
        with pytest.raises(ValueError, match=r"Unknown group key|foo"):
            parse_groups(foo={}, redshift=Fixed(0.1))

    def test_unknown_sfh_type_raises_value_error(self):
        """Unknown SFH type should raise ValueError."""
        with pytest.raises(ValueError, match=r"Unknown.*type.*sfh|banana"):
            parse_groups(
                sfh={"type": "banana"},
                redshift=Fixed(0.1),
            )

    def test_unknown_dust_type_raises_value_error(self):
        """Unknown dust type should raise ValueError."""
        with pytest.raises(ValueError, match=r"dust|magic"):
            parse_groups(
                dust_attenuation={"type": "magic"},
                redshift=Fixed(0.1),
            )

    def test_unknown_neb_type_raises_value_error(self):
        """Unknown nebular type should raise ValueError."""
        with pytest.raises(ValueError, match=r"nebular|invalid"):
            parse_groups(
                neb={"type": "invalid"},
                redshift=Fixed(0.1),
            )

    def test_unknown_igm_type_raises_value_error(self):
        """Unknown IGM type should raise ValueError."""
        with pytest.raises(ValueError, match=r"IGM|invalid"):
            parse_groups(
                igm={"type": "invalid"},
                redshift=Fixed(0.1),
            )


class TestTopLevel:
    """Test top-level kwargs (redshift, apply_igm, etc.)."""

    def test_redshift_fixed_value(self):
        """redshift=Fixed(0.05) should override to Fixed(0.05)."""
        params = parse_groups(
            redshift=Fixed(0.05),
        )
        dist = params.get_distribution("redshift")
        assert isinstance(dist, Fixed)
        assert dist.value == 0.05

    def test_redshift_uniform_prior(self):
        """redshift=Uniform(...) should make redshift free."""
        params = parse_groups(
            redshift=Uniform(0.01, 0.1),
        )
        assert "redshift" in params.free_params
        dist = params.get_distribution("redshift")
        assert isinstance(dist, Uniform)
        assert dist.lo == 0.01
        assert dist.hi == 0.1

    def test_redshift_bare_value_becomes_fixed(self):
        """redshift=0.1 should become Fixed(0.1)."""
        params = parse_groups(redshift=0.1)
        dist = params.get_distribution("redshift")
        assert isinstance(dist, Fixed)
        assert dist.value == 0.1

    def test_apply_igm_no_longer_overrides_the_group(self):
        """There is nothing left to override with.

        This used to assert that a top-level ``apply_igm=False`` beat an
        activating ``igm`` group -- the behavior of the secondary switch that
        has now been retired precisely because it could disagree with the group
        beside it. The group is the only statement of activation, so passing
        the old kwarg raises instead of quietly winning.
        """
        with pytest.raises(ValueError, match="apply_igm is retired"):
            parse_groups(
                igm={"type": "madau"},
                apply_igm=False,
                redshift=Fixed(0.1),
            )

    def test_igm_type_none_is_how_you_turn_it_off(self):
        """The replacement for the retired override."""
        params = parse_groups(igm={"type": "none"}, redshift=Fixed(0.1))
        assert params.apply_igm is False


class TestCanonicalExample:
    """Test the exact example from the spec."""

    def test_canonical_example_from_spec(self):
        """The canonical example must work end-to-end."""
        params = parse_groups(
            sfh={
                "type": "dpl",
                "*": FREE,
                "beta": Uniform(0.3, 2.0),  # Must be positive
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            dust_emission={"type": "dale2014", "*": FIXED},
            neb={"type": "cue", "*": FIXED},
            redshift=Uniform(0.01, 5.0),
        )

        assert isinstance(params, Parameters)
        assert "sfh_dpl_beta" in params.free_params
        assert "sfh_dpl_alpha" in params.free_params
        assert "dust_tau_bc" in params.fixed_params
        assert params.get_distribution("dust_tau_bc").value == 0.5
        assert params.dust_law_bc == "calzetti"
        assert params.nebular_mode == "cue"
        assert "redshift" in params.free_params


class TestFreeFixedSentinels:
    """Test FREE/FIXED sentinel behavior."""

    def test_free_sentinel_identity(self):
        """FREE sentinel should preserve identity across copy/pickle."""
        import copy

        sentinel = FREE
        copied = copy.deepcopy({sentinel})
        # Sentinel identity should be preserved
        assert FREE in copied or next(iter(copied)) is FREE

    def test_fixed_sentinel_identity(self):
        """FIXED sentinel should preserve identity across copy/pickle."""
        import copy

        sentinel = FIXED
        copied = copy.deepcopy({sentinel})
        # Sentinel identity should be preserved
        assert FIXED in copied or next(iter(copied)) is FIXED


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_empty_group_dict(self):
        """Empty group dict should use all defaults fixed."""
        params = parse_groups(
            sfh={"type": "dpl"},
            redshift=Fixed(0.1),
        )
        # All dpl params should be fixed
        for param in [
            "sfh_dpl_alpha",
            "sfh_dpl_beta",
            "sfh_dpl_tau_gyr",
            "sfh_dpl_log_total_mass",
        ]:
            assert param in params.fixed_params

    def test_no_groups_at_all(self):
        """Calling parse_groups with only top-level should work."""
        params = parse_groups(redshift=Fixed(0.1))
        assert isinstance(params, Parameters)
        assert "redshift" in params.all_params

    def test_full_prefix_override_under_star_fixed(self):
        """Issue #424: per-param override under '*': FIXED must accept the
        full-prefixed key (``neb_logU``) as well as the short form
        (``logU``). Previously the full-prefix form was silently dropped,
        leaving the default in place — a silent footgun for new users.
        """
        # Short form (always worked).
        short = parse_groups(
            neb={"type": "cue", "*": FIXED, "logU": Fixed(-2.5)},
            redshift=Fixed(0.01),
        )
        # Full-prefix form (regressed silently before #424).
        full = parse_groups(
            neb={"type": "cue", "*": FIXED, "neb_logU": Fixed(-2.5)},
            redshift=Fixed(0.01),
        )
        assert float(short._distributions["neb_logU"].value) == -2.5
        assert float(full._distributions["neb_logU"].value) == -2.5

    def test_full_prefix_override_in_sfh_group(self):
        """Full-prefixed SFH keys (``sfh_dpl_alpha``) also resolve."""
        params = parse_groups(
            sfh={
                "type": "dpl",
                "*": FIXED,
                "sfh_dpl_alpha": Fixed(7.0),
            },
            redshift=Fixed(0.1),
        )
        assert float(params._distributions["sfh_dpl_alpha"].value) == 7.0

    def test_multiple_dust_law_params(self):
        """dust with both law_bc and law_diff should work."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "smc",
                "*": FIXED,
            },
            redshift=Fixed(0.1),
        )
        assert params.dust_law_bc == "calzetti"
        assert params.dust_law_diff == "smc"


class TestCueOptionalKnobExposure:
    """#653: the nested-dict builder must expose Cue's optional knobs
    (gas density / abundances, ionizing-spectrum shape) for a ``type='cue'``
    neb group, not just logU / logZ_gas."""

    def test_gas_logn_settable_and_free(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={"type": "cue", "*": FIXED, "gas_logn": Uniform(1.0, 4.0)},
            redshift=Fixed(0.05),
        )
        assert "gas_logn" in params.free_params

    def test_ionspec_slopes_settable_and_free(self):
        """The ionizing-spectrum shape can be inferred (free ionspec_*)."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={
                "type": "cue",
                "*": FIXED,
                "ionspec_index1": Uniform(1.0, 20.0),
                "ionspec_logLratio1": Uniform(0.0, 5.0),
            },
            redshift=Fixed(0.05),
        )
        assert "ionspec_index1" in params.free_params
        assert "ionspec_logLratio1" in params.free_params

    def test_gas_abundances_settable(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={
                "type": "cue",
                "*": FIXED,
                "gas_logno": Uniform(-1.0, 1.0),
                "gas_logco": Fixed(0.1),
            },
            redshift=Fixed(0.05),
        )
        assert "gas_logno" in params.free_params
        assert "gas_logco" in params._distributions

    def test_bare_value_is_fixed(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={"type": "cue", "*": FIXED, "gas_logn": 2.0},
            redshift=Fixed(0.05),
        )
        assert "gas_logn" not in params.free_params
        assert "gas_logn" in params._distributions

    def test_optional_knob_rejected_for_non_cue(self):
        """gas_logn is a Cue-only knob; an ssp/cloudy neb group must reject it."""
        with pytest.raises(ValueError, match="gas_logn"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                neb={"type": "ssp", "gas_logn": Uniform(1.0, 4.0)},
                redshift=Fixed(0.05),
            )

    def test_stale_name_still_rejected(self):
        """The pre-fix gallery name ``neb_n_h`` is not a real param; reject it."""
        with pytest.raises(ValueError, match="neb_n_h"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                neb={"type": "cue", "neb_n_h": Fixed(2.0)},
                redshift=Fixed(0.05),
            )

    def test_free_sentinel_without_prior_raises(self):
        """Optional Cue knobs have no registry default, so a bare FREE/'*'
        cannot expand them — require an explicit prior."""
        with pytest.raises(ValueError, match="explicit prior"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                neb={"type": "cue", "*": FIXED, "gas_logn": FREE},
                redshift=Fixed(0.05),
            )


class TestElineModeExposure:
    """#653: eline_mode is a recognized top-level builder setting, so the
    line-velocity params register through the nested-dict path (previously
    only the flat Parameters(eline_mode=...) constructor saw them)."""

    def test_fitted_registers_eline_params(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={"type": "cue", "*": FIXED},
            eline_mode="fitted",
            redshift=Fixed(0.05),
        )
        assert "eline_sigma_kms" in params.all_params
        assert "eline_delta_v_kms" in params.all_params

    def test_off_registers_nothing(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={"type": "cue", "*": FIXED},
            redshift=Fixed(0.05),
        )
        assert "eline_sigma_kms" not in params.all_params


class TestAllParamsAlias:
    """The preferred ``all_params`` key is an exact synonym for the ``'*'``
    wildcard: it parses identically, works in sub-blocks, errors on the same
    bad values, and cannot coexist with ``'*'`` in the same dict.
    """

    @pytest.mark.parametrize("wildcard_key", ["*", "all_params"])
    def test_free_frees_all_params(self, wildcard_key):
        """``all_params: FREE`` frees every declared param, same as ``'*': FREE``."""
        params = parse_groups(
            sfh={"type": "dpl", wildcard_key: FREE},
            redshift=Fixed(0.1),
        )
        for p in ("sfh_dpl_alpha", "sfh_dpl_beta", "sfh_dpl_tau_gyr", "sfh_dpl_log_total_mass"):
            assert p in params.free_params

    @pytest.mark.parametrize("wildcard_key", ["*", "all_params"])
    def test_fixed_fixes_all_params(self, wildcard_key):
        """``all_params: FIXED`` fixes every declared param, same as ``'*': FIXED``."""
        params = parse_groups(
            sfh={"type": "dpl", wildcard_key: FIXED},
            redshift=Fixed(0.1),
        )
        for p in ("sfh_dpl_alpha", "sfh_dpl_beta", "sfh_dpl_tau_gyr", "sfh_dpl_log_total_mass"):
            assert p in params.fixed_params

    def test_alias_equivalent_to_star(self):
        """``all_params`` and ``'*'`` produce bit-identical free/fixed partitions."""
        common = dict(
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            neb={"type": "cue"},
            redshift=Uniform(0.01, 5.0),
        )
        star = parse_groups(
            sfh={"type": "dpl", "*": FREE, "beta": Fixed(1.5)},
            **{
                **common,
                "dust_attenuation": {"law": "power_law", **common["dust_attenuation"], "*": FIXED},
                "neb": {**common["neb"], "*": FIXED},
            },
        )
        alias = parse_groups(
            sfh={"type": "dpl", "all_params": FREE, "beta": Fixed(1.5)},
            **{
                **common,
                "dust_attenuation": {
                    "law": "power_law",
                    **common["dust_attenuation"],
                    "all_params": FIXED,
                },
                "neb": {**common["neb"], "all_params": FIXED},
            },
        )
        assert star.free_params == alias.free_params
        assert star.fixed_params == alias.fixed_params

    def test_per_param_override_beats_alias(self):
        """A per-parameter override still wins over ``all_params: FREE``."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": FREE, "beta": Fixed(2.0)},
            redshift=Fixed(0.1),
        )
        assert "sfh_dpl_beta" in params.fixed_params
        assert params.get_distribution("sfh_dpl_beta").value == 2.0
        assert "sfh_dpl_alpha" in params.free_params

    def test_alias_in_dust_emission_subblock(self):
        """``all_params`` resolves inside a nested sub-block (dust.emission),
        identically to ``'*'``."""

        def build(wk):
            return parse_groups(
                sfh={"type": "dpl", wk: FIXED},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    wk: FIXED,
                    # FIXED, not FREE: this test is about the alias resolving
                    # identically to '*', not about what the wildcard frees.
                },
                dust_emission={
                    "type": "dale2014",
                    wk: FIXED,
                    "alpha_dale": Uniform(0.5, 4.0),
                },
                redshift=Fixed(0.1),
            )

        star, alias = build("*"), build("all_params")
        assert star.free_params == alias.free_params
        assert "dust_alpha_dale" in alias.free_params

    def test_alias_block_scoped_in_agn(self):
        """A top-level ``agn={'all_params': FREE}`` is block-scoped just like ``'*'``."""
        star = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            agn={"type": "simple", "*": FREE},
            redshift=Fixed(0.1),
        )
        alias = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED},
            agn={"type": "simple", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        assert star.free_params == alias.free_params

    def test_alias_in_agn_composable_subblock(self):
        """``all_params`` resolves inside a composable AGN sub-block (agn.disc),
        identically to ``'*'``."""

        def build(wk):
            return parse_groups(
                sfh={"type": "dpl", wk: FIXED},
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor", wk: FREE},
                    "torus": {"type": "skirtor"},
                    "nlr": {"type": "none"},
                    "blr": {"type": "none"},
                },
                redshift=Fixed(0.1),
            )

        # Equivalence through the AGN sub-block merge/inheritance path
        # (_make_group_view) is the contract: all_params must be handled
        # identically to '*' at every nesting depth.
        star, alias = build("*"), build("all_params")
        assert star.free_params == alias.free_params
        assert star.fixed_params == alias.fixed_params

    def test_both_keys_present_raises(self):
        """Setting both ``'*'`` and ``all_params`` in one dict is ambiguous."""
        with pytest.raises(ValueError, match="wildcard once"):
            parse_groups(
                sfh={"type": "dpl", "*": FREE, "all_params": FIXED},
                redshift=Fixed(0.1),
            )

    def test_both_keys_present_raises_in_subblock(self):
        """The both-present guard also fires inside a nested sub-block."""
        with pytest.raises(ValueError, match="wildcard once"):
            parse_groups(
                sfh={"type": "dpl", "all_params": FIXED},
                dust_attenuation={
                    "type": "two_component",
                },
                dust_emission={"type": "dale2014", "*": FREE, "all_params": FIXED},
                redshift=Fixed(0.1),
            )

    @pytest.mark.parametrize("bad", ["free", "fixed", None, True])
    def test_invalid_alias_value_raises(self, bad):
        """A non-sentinel ``all_params`` value raises the same error as ``'*'``."""
        with pytest.raises(ValueError, match="all_params"):
            parse_groups(sfh={"type": "dpl", "all_params": bad}, redshift=Fixed(0.1))


@pytest.mark.regression_bug
class TestRoundTripGroupTypes:
    """Active radio / IGM selections must survive ``to_groups()``.

    ``_extract_group_type`` probed ``spec.radio_model`` — an attribute the
    parse side never sets (it stores ``radio`` + ``radio_sfr_mode`` +
    ``radio_agn_model``) — so the round-trip emitted a radio group with no
    type anywhere. IGM was worse: an active selection has no parameters of
    its own, and the no-params emission path only handled ``"none"``, so
    ``igm={'type': 'madau'}`` vanished and silently rebuilt as the default
    Inoue+2014 model. Same failure class as the ``ForwardModel._approx``
    migration miss: a hasattr probe for a never-set attribute fails open.
    """

    def _roundtrip(self, **kw):
        spec = parse_groups(redshift=Fixed(0.05), **kw)
        groups = spec.to_groups()
        return spec, groups, parse_groups(**groups)

    def test_radio_types_ride_on_the_sub_blocks(self):
        _, groups, rebuilt = self._roundtrip(
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}
        )
        radio = groups["radio"]
        assert radio["sf"]["type"] == "bell2003"
        assert radio["agn"]["type"] == "powerlaw"
        # Composable form never has a top-level 'type' key
        assert "type" not in radio
        assert rebuilt.radio is True

    def test_radio_composable_variant_survives(self):
        _, groups, rebuilt = self._roundtrip(radio={"sf": {"type": "delvecchio2021"}})
        assert groups["radio"]["sf"]["type"] == "delvecchio2021"
        assert rebuilt.radio_sfr_mode == "delvecchio2021"

    def test_igm_madau_is_not_silently_swapped_for_the_default(self):
        _, groups, rebuilt = self._roundtrip(igm={"type": "madau"})
        assert groups["igm"] == {"type": "madau"}
        assert rebuilt.igm_model == "madau"

    def test_igm_default_omitted_but_rebuild_matches(self):
        """The default model may be elided — as long as the rebuild agrees."""
        spec, _, rebuilt = self._roundtrip(igm={"type": "inoue14"})
        assert rebuilt.apply_igm is True
        assert rebuilt.igm_model == spec.igm_model

    def test_igm_none_round_trips_off(self):
        _, _, rebuilt = self._roundtrip(igm={"type": "none"})
        assert rebuilt.apply_igm is False
