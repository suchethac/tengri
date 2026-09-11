# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust attenuation key vocabulary and normalization.

This module tests the dust key vocabulary system introduced to centralize
the definition of per-screen stems (dust_slope, dust_delta, dust_Rv,
dust_bump_strength) and their screen suffixes (bc, diff, neb).

Coverage includes:
- Round-trip spelling conversions (full_to_short, short_to_full)
- Screen suffix detection (split_screen_suffix)
- Per-screen key generation and validation
- Single-home verification for the OVERRIDE_STEMS and TWO_COMPONENT_OVERRIDE_KEYS
- Grammar normalization (dust_tau_bc and tau_bc equivalence)
- Validation of per-screen keys against declared attenuation laws
"""

from __future__ import annotations

import pytest

from tengri import Fixed
from tengri.components.dust._apply import (
    _TWO_COMPONENT_LAW_PARAMS,
    TWO_COMPONENT_OVERRIDE_KEYS,
)
from tengri.components.dust.laws._registry import DUST_LAWS, law_kwarg_names
from tengri.config.exceptions import ParameterError
from tengri.parameters._dust_keys import (
    OVERRIDE_STEMS,
    full_to_short,
    normalize_dust_group_keys,
    per_screen_keys,
    short_to_full,
    split_screen_suffix,
)
from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS, parse_groups

pytestmark = pytest.mark.contract


class TestRoundTrip:
    """full_to_short and short_to_full are true round-trips."""

    @pytest.mark.parametrize(
        "stem",
        [*OVERRIDE_STEMS, "tau_bc", "tau_diff", "tau_v", "f_obscuration"],
    )
    def test_round_trip_preserves_stem(self, stem):
        """short_to_full then full_to_short gives back the stem."""
        full = short_to_full(stem)
        back = full_to_short(full)
        assert back == stem

    @pytest.mark.parametrize(
        "stem",
        [*OVERRIDE_STEMS, "tau_bc", "tau_diff", "tau_v", "f_obscuration"],
    )
    def test_full_name_always_starts_with_dust(self, stem):
        """short_to_full always returns a dust_ prefixed name."""
        full = short_to_full(stem)
        assert full.startswith("dust_"), f"{stem!r} -> {full!r}"

    def test_dust_prefix_is_idempotent(self):
        """short_to_full of an already-prefixed name is idempotent."""
        assert short_to_full("dust_slope") == "dust_slope"
        assert short_to_full("dust_delta") == "dust_delta"


class TestSplitScreenSuffix:
    """split_screen_suffix detects and strips screen suffixes."""

    def test_slope_bc_splits(self):
        """slope_bc splits into ('slope', 'bc')."""
        assert split_screen_suffix("slope_bc") == ("slope", "bc")

    def test_tau_diff_splits(self):
        """tau_diff splits into ('tau', 'diff')."""
        assert split_screen_suffix("tau_diff") == ("tau", "diff")

    def test_no_suffix_returns_none_screen(self):
        """bump_strength without a screen suffix returns (key, None)."""
        assert split_screen_suffix("bump_strength") == ("bump_strength", None)

    def test_unknown_suffix_returns_none(self):
        """delta_unknown returns (key, None) because 'unknown' is not a valid screen."""
        assert split_screen_suffix("delta_unknown") == ("delta_unknown", None)


class TestPerScreenKeys:
    """per_screen_keys derives the full set of per-screen keys."""

    def test_per_screen_keys_is_in_structural_keys(self):
        """Every per-screen key from per_screen_keys() is in _GROUP_STRUCTURAL_KEYS."""
        screens_set = per_screen_keys()
        structural = _GROUP_STRUCTURAL_KEYS["dust_attenuation"]
        assert screens_set <= structural, (
            f"Per-screen keys not in structural: {screens_set - structural}"
        )

    def test_structural_per_screen_keys_match_per_screen_keys(self):
        """The per-screen keys in structural (override stems × screens) equal per_screen_keys()."""
        structural = _GROUP_STRUCTURAL_KEYS["dust_attenuation"]
        per_screen = per_screen_keys()
        per_screen_from_structural = {
            k
            for k in structural
            if split_screen_suffix(k)[0] in OVERRIDE_STEMS
            and split_screen_suffix(k)[1] is not None
        }
        assert per_screen_from_structural == per_screen


class TestOneHome:
    """OVERRIDE_STEMS and TWO_COMPONENT_OVERRIDE_KEYS share one definition."""

    def test_override_keys_stem_set_matches(self):
        """set(TWO_COMPONENT_OVERRIDE_KEYS) == set(OVERRIDE_STEMS)."""
        assert set(TWO_COMPONENT_OVERRIDE_KEYS) == set(OVERRIDE_STEMS)

    def test_override_keys_map_to_full_names(self):
        """Every TWO_COMPONENT_OVERRIDE_KEYS[stem] == short_to_full(stem)."""
        for stem in OVERRIDE_STEMS:
            assert TWO_COMPONENT_OVERRIDE_KEYS[stem] == short_to_full(stem)

    def test_two_component_law_params_stems_match(self):
        """_TWO_COMPONENT_LAW_PARAMS keywords match OVERRIDE_STEMS (as full names)."""
        declared_full_names = {kw for kw, _flat, _d in _TWO_COMPONENT_LAW_PARAMS}
        expected = {short_to_full(s) for s in OVERRIDE_STEMS}
        assert declared_full_names == expected


class TestEveryLawKeywordDeclared:
    """Every law's kwargs are declared by dust attenuation parameters."""

    def test_every_law_keyword_in_declared_params(self):
        """Law keywords (minus wavelength/redshift) are a subset of declared dust params."""
        from tengri.components.dust._params import ATTENUATION_PARAMS, SINGLE_COMPONENT_PARAMS

        # Gather declared parameter names from dust component parameter declarations
        declared_names = {param.name for param in ATTENUATION_PARAMS}
        declared_names |= {param.name for param in SINGLE_COMPONENT_PARAMS}

        # Known undeclared kwargs that are read by laws but not declared yet.
        # dust_tea_scatter: read by tea_emission law; a follow-up commit declares it.
        # dust_c1, dust_c2, dust_c3, dust_c4: read by li08 law; similar follow-up commit.
        # dust_bump_x0, dust_bump_gamma: read by noll09 law; similar follow-up commit.
        _KNOWN_UNDECLARED = frozenset(
            {
                "dust_tea_scatter",
                "dust_c1",
                "dust_c2",
                "dust_c3",
                "dust_c4",
                "dust_bump_x0",
                "dust_bump_gamma",
            }
        )

        # For each law, its kwargs (minus wavelength and redshift) must be in declared
        for law_name in DUST_LAWS:
            kw_set = law_kwarg_names(law_name)
            # law_kwarg_names may or may not include wavelength/redshift; check both
            subset = kw_set - {"wavelength", "redshift"}
            missing = subset - declared_names - _KNOWN_UNDECLARED
            assert not missing, (
                f"Law {law_name!r} has undeclared kwargs: {missing}. "
                f"Declared: {declared_names}, Undeclared OK: {_KNOWN_UNDECLARED}"
            )


class TestNormalizeDustGroupKeys:
    """normalize_dust_group_keys strips leading dust_ prefix consistently."""

    def test_normalize_dust_prefix_to_stem(self):
        """dust_tau_bc, dust_law_bc keys normalize to their stems."""
        result = normalize_dust_group_keys(
            {"dust_tau_bc": 1, "law": "x", "dust_curve": "smc"},
            accepted={"tau_bc", "law", "dust_curve"},
        )
        assert result == {"tau_bc": 1, "law": "x", "dust_curve": "smc"}

    def test_normalize_unknown_dust_prefix_stays(self):
        """dust_bogus stays as-is because bogus is not accepted."""
        result = normalize_dust_group_keys({"dust_bogus": 1}, accepted={"tau_bc"})
        assert result == {"dust_bogus": 1}

    def test_normalize_non_dust_prefix_unchanged(self):
        """Keys that don't start with dust_ stay unchanged."""
        result = normalize_dust_group_keys(
            {"tau_bc": 1, "law": "calzetti"},
            accepted={"tau_bc", "law"},
        )
        assert result == {"tau_bc": 1, "law": "calzetti"}

    def test_normalize_two_spellings_raises(self):
        """Both tau_bc and dust_tau_bc present -> ValueError."""
        with pytest.raises(ValueError, match="two spellings"):
            normalize_dust_group_keys(
                {"tau_bc": 1, "dust_tau_bc": 2},
                accepted={"tau_bc"},
            )

    def test_normalize_non_dict_unchanged(self):
        """Non-dict input (list, string, etc.) returned unchanged."""
        assert normalize_dust_group_keys([1, 2, 3], accepted={"a"}) == [1, 2, 3]
        assert normalize_dust_group_keys("string", accepted={"a"}) == "string"

    def test_normalize_does_not_mutate_input(self):
        """The input dict is not mutated; a new dict is returned."""
        original = {"dust_tau_bc": 1}
        result = normalize_dust_group_keys(original, accepted={"tau_bc"})
        assert original == {"dust_tau_bc": 1}, "Input was mutated"
        assert result == {"tau_bc": 1}


class TestGrammarNormalization:
    """Grammar paths normalize dust_* spellings before passes see them."""

    def test_dust_tau_diff_normalized_in_two_component(self):
        """dust_attenuation with dust_tau_diff normalizes to tau_diff."""
        spec = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.5),
                "dust_tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.5),
        )
        # The internal representation should have tau_diff, not dust_tau_diff
        # We check this via the spec's get_distribution method
        dist = spec.get_distribution("dust_tau_diff")
        assert dist == Fixed(0.2)

    def test_dust_law_bc_normalized_in_two_component(self):
        """dust_law_bc normalizes to law_bc in the grammar."""
        spec = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "two_component",
                "dust_law_bc": "calzetti",
                "dust_law_diff": "power_law",
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.2),
            },
            redshift=Fixed(0.5),
        )
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "power_law"

    def test_dust_slope_bc_in_overrides(self):
        """dust_slope_bc normalizes and lands in dust_law_overrides."""
        spec = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "two_component",
                "law": "power_law",
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.2),
                "dust_slope_bc": -1.0,
                "dust_slope_diff": -0.5,
            },
            redshift=Fixed(0.5),
        )
        assert spec.dust_law_overrides == {
            "bc": {"dust_slope": -1.0},
            "diff": {"dust_slope": -0.5},
        }

    def test_wg00_dust_curve_unchanged(self):
        """wg00 with dust_curve builds without error (dust_curve is in structural keys)."""
        spec = parse_groups(
            sfh={"type": "dpl"},
            dust_attenuation={
                "type": "wg00",
                "dust_curve": "mw",
                "geometry": "shell",
                "structure": "homogeneous",
                "tau_v": Fixed(0.3),
            },
            redshift=Fixed(0.1),
        )
        assert spec.dust_model == "wg00"

    def test_delta_bc_unread_by_law_raises(self):
        """delta_bc under calzetti (which doesn't read delta) raises."""
        with pytest.raises(ParameterError, match="'delta_bc'"):
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "tau_bc": Fixed(0.5),
                    "tau_diff": Fixed(0.2),
                    "delta_bc": -0.3,
                },
                redshift=Fixed(0.5),
            )

    def test_duplicate_spellings_raise(self):
        """tau_bc and dust_tau_bc in the same dict -> ValueError."""
        with pytest.raises(ValueError, match="two spellings"):
            parse_groups(
                sfh={"type": "dpl"},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "tau_bc": Fixed(0.5),
                    "dust_tau_bc": Fixed(0.5),
                    "tau_diff": Fixed(0.2),
                },
                redshift=Fixed(0.5),
            )
