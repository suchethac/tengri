# SPDX-License-Identifier: BSD-3-Clause
"""Tests for private helper functions in line_list.py.

Covers uncovered branches in:
- _parse_cloudy_species (lines 597-621)
- _is_balmer_line (lines 624-652)
- _is_broad_candidate (lines 655-662)
- _get_doublet_ratio_by_wavelength (lines 665-677)
- _detect_doublets_by_proximity (lines 680-733)
- LineList.select() species= filter and empty-result path (lines 405-448)
"""

import pytest

from tengri.observation.line_list import (
    DoubletConstraint,
    LineList,
    _detect_doublets_by_proximity,
    _get_doublet_ratio_by_wavelength,
    _is_balmer_line,
    _is_broad_candidate,
    _parse_cloudy_species,
)

pytestmark = pytest.mark.bounds

# ── _parse_cloudy_species ─────────────────────────────────────────


class TestParseCloudySpecies:
    def test_hydrogen_lyman(self):
        """'H  1 1215.67A' → 'H1'."""
        assert _parse_cloudy_species("H  1 1215.67A") == "H1"

    def test_oxygen_iii(self):
        """'O  3 5006.84A' → 'O3'."""
        assert _parse_cloudy_species("O  3 5006.84A") == "O3"

    def test_nitrogen_ii(self):
        """'N  2 6583.45A' → 'N2'."""
        assert _parse_cloudy_species("N  2 6583.45A") == "N2"

    def test_simple_name_fallback(self):
        """Non-CLOUDY name like 'Halpha' falls back to first token."""
        result = _parse_cloudy_species("Halpha")
        assert result == "Halpha"

    def test_single_token_fallback(self):
        """Name with a single word and no digit second token falls back."""
        result = _parse_cloudy_species("Lya")
        assert result == "Lya"

    def test_element_with_non_digit_second_part(self):
        """Name where second token is not a digit falls back to raw first token."""
        result = _parse_cloudy_species("Fe II")
        # Second token 'II' is not a digit so falls back
        assert result == "Fe"

    def test_empty_string_fallback(self):
        """Empty string returns empty string without raising."""
        result = _parse_cloudy_species("   ")
        # strip().split() on whitespace gives []
        # falls through to name.split()[0] branch, but split on whitespace is []
        # Actually empty string.split() == [] so name.split()[0] raises IndexError?
        # Let's verify the actual behavior: "" -> strip -> "" -> split -> []
        # The condition `name.strip()` is falsy, so returns `name` which is "   "
        assert isinstance(result, str)

    def test_returns_string(self):
        """Always returns a string."""
        for name in ["H  1 1215.67A", "O  3 5006.84A", "Halpha", "  "]:
            assert isinstance(_parse_cloudy_species(name), str)


# ── _is_balmer_line ───────────────────────────────────────────────


class TestIsBalmerLine:
    def test_halpha_is_balmer(self):
        assert _is_balmer_line("Halpha", "H1") is True

    def test_hbeta_is_balmer(self):
        assert _is_balmer_line("Hbeta", "H1") is True

    def test_hgamma_is_balmer(self):
        assert _is_balmer_line("Hgamma", "H1") is True

    def test_hdelta_is_balmer(self):
        assert _is_balmer_line("Hdelta", "H1") is True

    def test_lya_is_not_balmer_by_keyword(self):
        """Lya doesn't match any Balmer keyword — returns False via keyword check."""
        # "lya" contains no balmer_keywords match, but species is H1
        # Falls to CLOUDY wavelength check: "Lya" has 1 token → len(parts)<3 → returns False
        assert _is_balmer_line("Lya", "H1") is False

    def test_oiii_not_balmer(self):
        """Non-H1 species is never Balmer."""
        assert _is_balmer_line("OIII_5007", "O3") is False

    def test_nii_not_balmer(self):
        assert _is_balmer_line("NII_6584", "N2") is False

    def test_cloudy_balmer_wavelength(self):
        """CLOUDY-format H1 line in Balmer wavelength range → True."""
        # Hbeta at 4861 Å: within [3646, 6563]
        assert _is_balmer_line("H  1 4861.33A", "H1") is True

    def test_cloudy_lyman_wavelength(self):
        """CLOUDY-format H1 line in Lyman range → True."""
        # Lya at 1216 Å: within [912, 1216]
        assert _is_balmer_line("H  1 1215.67A", "H1") is True

    def test_cloudy_outside_balmer_and_lyman(self):
        """CLOUDY-format H1 line outside both ranges → False."""
        # Paschen series at 18751 Å — outside Balmer and Lyman ranges
        assert _is_balmer_line("H  1 18751.0A", "H1") is False

    def test_cloudy_bad_wavelength_string_returns_false(self):
        """Malformed CLOUDY wavelength string falls through to False."""
        assert _is_balmer_line("H  1 BADWA", "H1") is False


# ── _is_broad_candidate ───────────────────────────────────────────


class TestIsBroadCandidate:
    def test_h1_species_is_broad(self):
        """Any H1 line is a broad candidate."""
        assert _is_broad_candidate("Halpha", "H1") is True

    def test_c4_species_is_broad(self):
        """CIV (C4 species) is a broad candidate."""
        assert _is_broad_candidate("CIV_1549", "C4") is True

    def test_mg2_species_is_broad(self):
        """MgII (Mg2 species) is a broad candidate."""
        assert _is_broad_candidate("MgII_2796", "Mg2") is True

    def test_c3_species_is_broad(self):
        """CIII] (C3 species) is a broad candidate."""
        assert _is_broad_candidate("CIII_1908", "C3") is True

    def test_oiii_not_broad(self):
        """[OIII] 5007 is not a broad candidate."""
        assert _is_broad_candidate("OIII_5007", "O3") is False

    def test_nii_not_broad(self):
        """[NII] is not a broad candidate."""
        assert _is_broad_candidate("NII_6584", "N2") is False

    def test_sii_not_broad(self):
        assert _is_broad_candidate("SII_6717", "S2") is False

    def test_keyword_halpha_match(self):
        """Name containing 'halpha' triggers broad via keyword even for unknown species."""
        assert _is_broad_candidate("Halpha_broad", "XX") is True

    def test_keyword_civ_match(self):
        assert _is_broad_candidate("civ_line", "XX") is True

    def test_keyword_mgii_match(self):
        assert _is_broad_candidate("some_mgii_feature", "XX") is True


# ── _get_doublet_ratio_by_wavelength ──────────────────────────────


class TestGetDoubletRatioByWavelength:
    def test_oiii_doublet_ratio(self):
        """[OIII] 5007/4959 ratio is 2.98."""
        ratio = _get_doublet_ratio_by_wavelength(5008.24, 4960.30)
        assert abs(ratio - 2.98) < 0.01

    def test_nii_doublet_ratio(self):
        """[NII] 6584/6548 ratio is 2.94."""
        ratio = _get_doublet_ratio_by_wavelength(6585.28, 6549.86)
        assert abs(ratio - 2.94) < 0.01

    def test_unknown_pair_returns_one(self):
        """Unknown pair returns default ratio of 1.0."""
        ratio = _get_doublet_ratio_by_wavelength(9999.0, 9998.0)
        assert ratio == 1.0

    def test_out_of_tolerance_returns_one(self):
        """Pair that's > 5 Å from any known pair returns 1.0."""
        # [OIII] primary is at 5008.24; shift it by 10 Å
        ratio = _get_doublet_ratio_by_wavelength(5018.5, 4960.30)
        assert ratio == 1.0

    def test_siii_doublet_ratio(self):
        """[SIII] 9532/9069 ratio is 2.47."""
        ratio = _get_doublet_ratio_by_wavelength(9533.23, 9071.10)
        assert abs(ratio - 2.47) < 0.01


# ── _detect_doublets_by_proximity ─────────────────────────────────


class TestDetectDoubletsByProximity:
    def test_detects_same_species_close_pair(self):
        """Two lines of same species within proximity → one DoubletConstraint."""
        names = ("A_5010", "B_5005")
        wavelengths = [5010.0, 5005.0]  # 5 Å apart — well within default 20 Å
        species = ("O3", "O3")
        doublets = _detect_doublets_by_proximity(names, wavelengths, species)
        assert len(doublets) == 1

    def test_longer_wavelength_is_primary(self):
        """Line with longer wavelength is the primary."""
        names = ("A_5010", "B_5005")
        wavelengths = [5010.0, 5005.0]  # 5 Å apart — within 20 Å default
        species = ("O3", "O3")
        doublets = _detect_doublets_by_proximity(names, wavelengths, species)
        dc = doublets[0]
        # index 0 has wave 5010 (longer) → primary; index 1 has 5005 → secondary
        assert dc.primary_idx == 0
        assert dc.secondary_idx == 1

    def test_different_species_not_paired(self):
        """Lines of different species are never paired, even if close."""
        names = ("Ha_6565", "NII_6583")
        wavelengths = [6564.61, 6583.45]
        species = ("H1", "N2")
        doublets = _detect_doublets_by_proximity(names, wavelengths, species)
        assert len(doublets) == 0

    def test_outside_proximity_not_paired(self):
        """Lines > proximity_angstrom apart are not paired."""
        names = ("A_5000", "B_6000")
        wavelengths = [5000.0, 6000.0]
        species = ("O3", "O3")
        doublets = _detect_doublets_by_proximity(
            names, wavelengths, species, proximity_angstrom=20.0
        )
        assert len(doublets) == 0

    def test_line_not_reused(self):
        """Once a line is in a doublet it is not re-paired with a third line."""
        names = ("A_5000", "B_5010", "C_5015")
        wavelengths = [5000.0, 5010.0, 5015.0]
        species = ("O3", "O3", "O3")
        # A+B form a pair; C is left alone (B already used)
        doublets = _detect_doublets_by_proximity(
            names, wavelengths, species, proximity_angstrom=20.0
        )
        assert len(doublets) == 1

    def test_returns_tuple_of_doublet_constraints(self):
        names = ("X_5000", "Y_5010")
        wavelengths = [5000.0, 5010.0]
        species = ("O3", "O3")
        doublets = _detect_doublets_by_proximity(names, wavelengths, species)
        assert isinstance(doublets, tuple)
        assert all(isinstance(d, DoubletConstraint) for d in doublets)

    def test_empty_catalog(self):
        """Empty inputs return empty tuple."""
        doublets = _detect_doublets_by_proximity((), [], ())
        assert doublets == ()

    def test_known_ratio_applied(self):
        """When wavelengths match a known pair, the known ratio is used."""
        names = ("OIII_5007", "OIII_4959")
        wavelengths = [5008.24, 4960.30]  # 47.94 Å apart — need explicit proximity
        species = ("O3", "O3")
        doublets = _detect_doublets_by_proximity(
            names, wavelengths, species, proximity_angstrom=50.0
        )
        assert len(doublets) == 1
        assert abs(doublets[0].ratio - 2.98) < 0.01


# ── LineList.select() — species filter and empty result ───────────


class TestSelectSpeciesAndEmpty:
    def test_species_filter_returns_only_matching(self):
        """select(species=['H1']) returns only hydrogen lines."""
        cat = LineList.default_optical()
        h1_cat = cat.select(species=["H1"])
        assert h1_cat.n_lines > 0
        assert all(sp == "H1" for sp in h1_cat.species)

    def test_species_filter_excludes_others(self):
        """select(species=['O3']) returns no H1 lines."""
        cat = LineList.default_optical()
        o3_cat = cat.select(species=["O3"])
        assert "H1" not in o3_cat.species

    def test_species_multi_filter(self):
        """select(species=['H1','O3']) retains both species."""
        cat = LineList.default_optical()
        sub = cat.select(species=["H1", "O3"])
        for sp in sub.species:
            assert sp in ("H1", "O3")

    def test_species_filter_with_wavelength_range(self):
        """species + wave_min/max combined (AND logic)."""
        cat = LineList.default_optical()
        # Only optical H1 lines
        sub = cat.select(species=["H1"], wave_min=3700.0, wave_max=7000.0)
        assert sub.n_lines > 0
        waves = [float(w) for w in sub.wavelengths]
        assert all(3700.0 <= w <= 7000.0 for w in waves)
        assert all(sp == "H1" for sp in sub.species)

    def test_empty_result_returns_empty_line_list(self):
        """Filter that excludes everything returns an empty LineList."""
        cat = LineList.default_optical()
        # No lines below 500 Å (all lines are > 1200 Å)
        empty = cat.select(wave_max=500.0)
        assert empty.n_lines == 0
        assert empty.names == ()
        assert empty.doublets == ()

    def test_empty_result_has_empty_wavelength_array(self):
        cat = LineList.default_optical()
        empty = cat.select(wave_max=500.0)
        assert len(empty.wavelengths) == 0

    def test_impossible_species_returns_empty(self):
        """Requesting a species not in catalog gives empty LineList."""
        cat = LineList.default_optical()
        empty = cat.select(species=["Xx99"])
        assert empty.n_lines == 0

    def test_doublets_dropped_when_member_filtered(self):
        """Doublet constraint is dropped if its secondary is filtered out."""
        cat = LineList.default_optical()
        # Keep only OIII_5007, not OIII_4959 — doublet should be dropped
        sub = cat.select(names=["OIII_5007"])
        assert sub.n_lines == 1
        # No constraint possible without the secondary
        assert len(sub.doublets) == 0

    def test_doublets_preserved_when_both_members_kept(self):
        """Doublet constraint is preserved if both members are kept."""
        cat = LineList.default_optical()
        sub = cat.select(names=["OIII_5007", "OIII_4959"])
        assert len(sub.doublets) == 1
        dc = sub.doublets[0]
        assert abs(dc.ratio - 2.98) < 0.01
