# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Parameters.to_groups() roundtrip.

Verifies that Parameters.to_groups() correctly inverts parse_groups(),
preserving all parameter distributions and structural choices.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract
from tengri.parameters import FIXED, FREE, Fixed, Uniform, parse_groups
from tengri.parameters.parameters import Parameters


def test_roundtrip_with_nebular_off():
    """Regression: to_groups must emit 'none' (not 'off') for disabled nebular.

    spec.nebular_mode returns 'off' when the nebular backend is disabled, but
    the parser's _VALID_NEBULAR_TYPES uses 'none'. Without translation, the
    round-trip would error: ``Unknown nebular type 'off'``.
    """
    orig = parse_groups(
        sfh={"type": "dpl", "*": FIXED},
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
            sfh={"type": "dpl", "*": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert isinstance(result, dict)

    def test_to_groups_contains_sfh_group(self):
        """to_groups() includes 'sfh' key when SFH is configured."""
        spec = parse_groups(
            sfh={"type": "dpl", "*": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "sfh" in result
        assert isinstance(result["sfh"], dict)

    def test_to_groups_contains_redshift_toplevel(self):
        """to_groups() includes 'redshift' at top level."""
        spec = parse_groups(
            sfh={"type": "dpl", "*": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "redshift" in result

    def test_to_groups_sfh_has_type_key(self):
        """to_groups() includes 'type' key in SFH group."""
        spec = parse_groups(
            sfh={"type": "dpl", "*": FREE},
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "type" in result["sfh"]
        assert result["sfh"]["type"] == "dpl"

    def test_to_groups_dust_nested_structure(self):
        """to_groups() preserves nested dust.emission subgroup structure."""
        spec = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "emission": {"type": "dale2014", "*": FIXED},
            },
            redshift=Fixed(0.1),
        )
        result = spec.to_groups()
        assert "dust" in result
        assert "emission" in result["dust"]
        assert result["dust"]["emission"]["type"] == "dale2014"


class TestToGroupsRoundtrip:
    """Roundtrip tests: from_groups -> to_groups -> from_groups."""

    def test_round_trip_minimal_dpl(self):
        """Minimal DPL model roundtrips with identical free/fixed sets."""
        original = parse_groups(
            sfh={"type": "dpl", "*": FREE},
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
            sfh={"type": "dpl", "*": FREE, "beta": Uniform(1, 3)},
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
            sfh={"type": "dpl", "*": FIXED},
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
                # FIXED, not FREE: FREE frees nothing on dale2014 and is now
                # refused. The round-trip property under test is unaffected.
                "emission": {"type": "dale2014", "*": FIXED},
            },
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
            sfh={"type": "dpl", "*": FIXED},
            dust={"type": "two_component", "*": FIXED},
            agn={
                "disc": {"type": "powerlaw", "*": FREE},
                "torus": {"type": "simple", "*": FIXED},
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
                "*": FREE,
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
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
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
        """When '*': FREE was used, those params should NOT appear explicitly (#1796).

        However, met_* params get implicit FIXED (no met block), creating a mix of
        wildcard_free and wildcard_fixed provenances that prevents full wildcard
        collapsing. This is the correct behavior: the roundtrip shows that met_*
        are Fixed while sfh_* are Free.
        """
        original = parse_groups(
            sfh={"type": "dpl", "*": FREE},
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
            sfh={"type": "dpl", "*": FREE, "beta": Uniform(1, 3)},
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
        assert "dust" in result
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
        """dust_law_bc setting is preserved."""
        original = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            dust={"type": "two_component", "law_bc": "kriek_conroy", "*": FIXED},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["dust"]["law_bc"] == "kriek_conroy"

    def test_to_groups_preserves_dust_emission_type(self):
        """dust_emission type is preserved."""
        original = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            dust={
                "type": "two_component",
                "*": FIXED,
                "emission": {"type": "dale2014", "*": FIXED},
            },
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["dust"]["emission"]["type"] == "dale2014"

    def test_to_groups_preserves_nebular_type(self):
        """nebular type is preserved."""
        original = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={"type": "cue", "*": FIXED},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["neb"]["type"] == "cue"

    def test_to_groups_preserves_apply_igm(self):
        """apply_igm setting is preserved."""
        original = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            redshift=Fixed(0.1),
            apply_igm=False,
        )
        result = original.to_groups()

        assert result["apply_igm"] is False

    def test_to_groups_preserves_sfh_composition(self):
        """SFH composition list is preserved."""
        original = parse_groups(
            sfh={"type": ["dpl", "field"], "*": FIXED},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        assert result["sfh"]["type"] == ["dpl", "field"]


class TestToGroupsEdgeCases:
    """Edge cases and special scenarios."""

    def test_to_groups_with_none_nebular(self):
        """to_groups works with nebular disabled (type='none')."""
        original = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
        result = original.to_groups()

        # When nebular is 'none', it's internally 'off'. Roundtrip should work.
        roundtripped = parse_groups(**result)
        assert roundtripped.nebular_mode == "off"
        assert original.nebular_mode == roundtripped.nebular_mode
