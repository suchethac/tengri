# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Parameters.to_groups() roundtrip.

Verifies that Parameters.to_groups() correctly inverts parse_groups(),
preserving all parameter distributions and structural choices.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract
from tengri.parameters import DEFAULT, FREE, Fixed, Uniform, parse_groups
from tengri.parameters.parameters import Parameters


def test_roundtrip_with_nebular_off():
    """Regression: to_groups must emit 'none' (not 'off') for disabled nebular.

    spec.nebular_mode returns 'off' when the nebular backend is disabled, but
    the parser's _VALID_NEBULAR_TYPES uses 'none'. Without translation, the
    round-trip would error: ``Unknown nebular type 'off'``.
    """
    orig = parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    assert orig.nebular_mode == "off"
    groups = orig.to_groups()
    neb = groups.get("neb", None)
    if neb is not None:
        assert neb.get("type") == "none", f"Expected 'none', got {neb.get('type')!r}"
    # Round-trip must not raise
    rebuilt = parse_groups(**groups)
    assert rebuilt.nebular_mode == "off"


class TestToGroupsBasic:
    """Basic structure and shape tests for to_groups()."""

    def test_to_groups_returns_dict(self):
        """to_groups() returns a dict."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert isinstance(result, dict)

    def test_to_groups_contains_sfh_group(self):
        """to_groups() includes 'sfh' key when SFH is configured."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "sfh" in result
        assert isinstance(result["sfh"], dict)

    def test_to_groups_contains_redshift_toplevel(self):
        """to_groups() includes 'redshift' at top level."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "redshift" in result

    def test_to_groups_sfh_has_type_key(self):
        """to_groups() includes 'type' key in SFH group."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "type" in result["sfh"]
        assert result["sfh"]["type"] == "dpl"

    def test_to_groups_dust_nested_structure(self):
        """to_groups() preserves nested dust.emission subgroup structure."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "dust_attenuation" in result
        # emission is a peer group now, not a sub-block of dust
        assert "emission" not in result["dust_attenuation"]
        assert result["dust_emission"]["type"] == "dale2014"


class TestToGroupsRoundtrip:
    """Roundtrip tests: from_groups -> to_groups -> from_groups."""

    def test_round_trip_minimal_dpl(self):
        """Minimal DPL model roundtrips with identical free/fixed sets."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        roundtripped = parse_groups(**original.to_groups())

        # Free and fixed sets must be identical
        assert set(original.free_params) == set(roundtripped.free_params)
        assert set(original.fixed_params) == set(roundtripped.fixed_params)

        # Each distribution must match
        for name in original.free_params:
            orig_dist = original.get_distribution(name)
            round_dist = roundtripped.get_distribution(name)
            assert orig_dist == round_dist, f"Mismatch for {name}: {orig_dist} vs {round_dist}"

        for name in original.fixed_params:
            orig_dist = original.get_distribution(name)
            round_dist = roundtripped.get_distribution(name)
            assert orig_dist == round_dist, f"Mismatch for {name}: {orig_dist} vs {round_dist}"

    def test_round_trip_with_explicit_override(self):
        """Explicit per-param overrides survive roundtrip."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1, 3)},
            redshift=Fixed(0.05),
        )
        roundtripped = parse_groups(**original.to_groups())

        assert original.free_params == roundtripped.free_params
        assert original.fixed_params == roundtripped.fixed_params

        # Beta override should survive
        orig_beta = original.get_distribution("sfh_dpl_beta")
        round_beta = roundtripped.get_distribution("sfh_dpl_beta")
        assert orig_beta == round_beta

    def test_round_trip_with_dust_emission(self):
        """Nested dust.emission sub-block roundtrips."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": 0.5,
                # Fixed(DEFAULT), not FREE: FREE frees nothing on dale2014 and is now
                # refused. The round-trip property under test is unaffected.
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        roundtripped = parse_groups(**original.to_groups())

        assert original.free_params == roundtripped.free_params
        assert original.fixed_params == roundtripped.fixed_params

        # Check a dust emission param survived
        orig_umin = original.get_distribution("dust_umin")
        round_umin = roundtripped.get_distribution("dust_umin")
        assert orig_umin == round_umin

    def test_round_trip_with_agn_composable(self):
        """AGN composable with sub-blocks roundtrips."""
        pytest.importorskip("grahsp", minversion=None)  # Skip if not available

        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "disc": {"type": "powerlaw", "all_params": FREE},
                "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
                "nlr": {"type": "analytic"},
                "blr": {"type": "none"},
            },
            redshift=Fixed(0.1),
        )
        roundtripped = parse_groups(**original.to_groups())

        assert original.free_params == roundtripped.free_params
        assert original.fixed_params == roundtripped.fixed_params

    def test_round_trip_with_sfh_composition(self):
        """SFH composition (list of types) roundtrips."""
        original = parse_groups(
            sfh={
                "type": ["dpl", "field"],
                "all_params": FREE,
            },
            redshift=Fixed(0.1),
        )
        roundtripped = parse_groups(**original.to_groups())

        assert original.free_params == roundtripped.free_params
        assert original.fixed_params == roundtripped.fixed_params

    def test_round_trip_mixed_free_fixed(self):
        """Mixed free and fixed params roundtrip."""
        original = parse_groups(
            sfh={"type": "dpl", "alpha": FREE, "beta": Uniform(0.5, 2.0), "tau_gyr": Fixed(1.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": Uniform(0, 1),
            },
            redshift=FREE,
        )
        roundtripped = parse_groups(**original.to_groups())

        assert original.free_params == roundtripped.free_params
        assert original.fixed_params == roundtripped.fixed_params

        for name in original.all_params:
            orig_dist = original.get_distribution(name)
            round_dist = roundtripped.get_distribution(name)
            assert orig_dist == round_dist, f"Mismatch for {name}"


class TestToGroupsWildcardCollapse:
    """Test wildcard collapsing logic."""

    def test_to_groups_omits_wildcard_expanded_params(self):
        """When 'all_params': FREE was used, those params should NOT appear explicitly (#1796).

        However, met_* params are implicitly Fixed (no met block), creating a mix of
        wildcard_free and wildcard_fixed provenances that prevents full wildcard
        collapsing. This is the correct behavior: the roundtrip shows that met_*
        are Fixed while sfh_* are Free.
        """
        original = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        sfh_dict = result["sfh"]
        # When there's no met block, met_* params are implicitly Fixed. This creates
        # a mix of wildcard provenances (sfh_* are wildcard_free, met_* are
        # wildcard_fixed), so the roundtrip can't collapse into a single wildcard.
        # The explicit listing correctly shows met_* as Fixed.
        assert "type" in sfh_dict
        # 'all_params' is NOT present because of the mixed wildcard types
        assert "all_params" not in sfh_dict
        # met_* params are listed explicitly showing they're Fixed
        assert "logzsol" in sfh_dict  # met_logzsol → short form 'logzsol'
        assert sfh_dict["logzsol"].is_fixed
        # sfh_* params are also listed explicitly (no wildcard collapse possible)
        assert "alpha" in sfh_dict  # sfh_dpl_alpha → short form 'alpha'

    def test_to_groups_preserves_user_overrides(self):
        """Explicit per-param overrides are preserved in the output dict."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1, 3)},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        sfh_dict = result["sfh"]
        # 'beta' was explicitly overridden, so it should be in the dict
        assert "beta" in sfh_dict
        assert sfh_dict["beta"] == Uniform(1, 3)

    def test_to_groups_no_wildcard_all_explicit(self):
        """When no wildcard was used, all params are explicit."""
        original = parse_groups(
            sfh={
                "type": "dpl",
                "alpha": FREE,
                "beta": Uniform(1, 3),
                "tau_gyr": Fixed(2.0),
                "log_total_mass": Uniform(8, 12),
            },
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        sfh_dict = result["sfh"]
        # DPL params should be explicit
        # met_logzsol and met_alpha_fe stay in the sfh group when
        # met_mode == "delta" (default) — only specs that opt into a
        # non-delta chemical-evolution mode (#311) emit a stellar block.
        # ``age_gyr`` (formation anchor, #514) joined the dpl param set; it
        # round-trips with its registry default even when left unspecified.
        dpl_keys = {"alpha", "beta", "tau_gyr", "age_gyr", "log_total_mass"}
        # ``logzsol_scatter`` (lognormal MDF width, #506) is a delta-mode
        # metallicity param, so like ``logzsol``/``alpha_fe`` it round-trips in
        # the sfh group until a non-delta chemical-evolution mode is selected.
        met_keys = {"logzsol", "alpha_fe", "logzsol_scatter"}
        expected_keys = {"type"} | dpl_keys | met_keys
        assert set(sfh_dict.keys()) == expected_keys


class TestToGroupsFlatBuilt:
    """Test to_groups on Parameters built via flat kwargs (no provenance)."""

    def test_to_groups_flat_built_has_no_wildcard(self):
        """Parameters built via flat kwargs have no wildcard, all params explicit."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_total_mass=Uniform(8, 12),
            dust_tau_bc=Uniform(0, 4),
            dust_tau_diff=Uniform(0, 3),
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()

        # Should have sfh, dust, redshift
        assert "sfh" in result
        assert "dust_attenuation" in result
        assert "redshift" in result

        # SFH should have type and all params explicit (no wildcard)
        sfh_dict = result["sfh"]
        assert sfh_dict["type"] == "dpl"
        assert "alpha" in sfh_dict
        assert "beta" in sfh_dict
        assert "tau_gyr" in sfh_dict
        assert "log_total_mass" in sfh_dict
        assert "*" not in sfh_dict, "No wildcard should be present"
        assert "all_params" not in sfh_dict, "No wildcard should be present"

    def test_round_trip_flat_built(self):
        """Flat-built Parameters roundtrip correctly."""
        original = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_total_mass=Uniform(8, 12),
            dust_tau_bc=Uniform(0, 2),
            redshift=Fixed(0.1),
        )
        roundtripped = parse_groups(**original.to_groups())

        assert original.free_params == roundtripped.free_params
        assert original.fixed_params == roundtripped.fixed_params


class TestToGroupsStructuralSettings:
    """Test that structural settings are preserved in to_groups output."""

    def test_to_groups_preserves_dust_law(self):
        """Differing per-screen laws are preserved as the law_bc/law_diff pair."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "kriek_conroy",
                "law_diff": "smc",
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["dust_attenuation"]["law_bc"] == "kriek_conroy"
        assert result["dust_attenuation"]["law_diff"] == "smc"

    def test_to_groups_collapses_shared_dust_law(self):
        """Equal per-screen laws round-trip as the shared 'law' key."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "kriek_conroy",
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["dust_attenuation"]["law"] == "kriek_conroy"
        assert "law_bc" not in result["dust_attenuation"]

    def test_to_groups_preserves_dust_emission_type(self):
        """dust_emission type is preserved."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["dust_emission"]["type"] == "dale2014"

    def test_to_groups_preserves_nebular_type(self):
        """nebular type is preserved."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["neb"]["type"] == "cue"

    def test_to_groups_preserves_igm_being_off(self):
        """The off state survives the round-trip -- through the group, not a flag.

        This used to assert ``result["apply_igm"] is False``. The emitter did
        emit that key, and it is now the one key its own parser refuses, so
        every round-trip raised on the way back in. Activation lives on the igm
        group, so the durable assertion is that reparsing the emitted dict
        gives back a model with IGM off -- which holds whether the group comes
        back as ``type: "none"`` or is omitted entirely, those now meaning the
        same thing.
        """
        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            igm={"type": "none"},
        )
        result = original.to_groups()

        assert "apply_igm" not in result, "the retired switch must not be emitted"
        assert parse_groups(**result).apply_igm is False

    def test_to_groups_preserves_sfh_composition(self):
        """SFH composition list is preserved."""
        original = parse_groups(
            sfh={"type": ["dpl", "field"], "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["sfh"]["type"] == ["dpl", "field"]


class TestToGroupsEdgeCases:
    """Edge cases and special scenarios."""

    def test_to_groups_with_none_nebular(self):
        """to_groups works with nebular disabled (type='none')."""
        original = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        # When nebular is 'none', it's internally 'off'. Roundtrip should work.
        roundtripped = parse_groups(**result)
        assert roundtripped.nebular_mode == "off"
        assert original.nebular_mode == roundtripped.nebular_mode
