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

# ── three predicate helpers, as tables ────────────────────────────
#
# 26 one-assertion tests become 3 parametrized ones. Each was a single call and
# a single `assert ... is True/False`, differing only in the argument -- the
# shape a table expresses better than 26 function definitions, and each case
# keeps its own id so a failure still names exactly which input broke.
#
# Two of the 26 did not survive as-is:
#
# * `test_empty_string_fallback` asserted only `isinstance(result, str)`, under
#   a docstring that reasoned aloud without reaching an answer ("Actually empty
#   string.split() == [] so name.split()[0] raises IndexError? Let's verify the
#   actual behavior"). Measured: whitespace-in is returned unchanged, and `""`
#   returns `""`. Both are pinned below, so the question the comment was asking
#   is now answered by the test.
# * `test_returns_string` looped over four names asserting each result was a
#   `str`. Every case below asserts exact string equality, which implies it.


#: (input, expected species). CLOUDY format is "<element> <ion> <wavelength>A";
#: anything else falls back to the first whitespace token, and a blank or
#: whitespace-only name is returned unchanged rather than raising.
_SPECIES_CASES = [
    ("H  1 1215.67A", "H1"),
    ("O  3 5006.84A", "O3"),
    ("N  2 6583.45A", "N2"),
    ("Halpha", "Halpha"),
    ("Lya", "Lya"),
    ("Fe II", "Fe"),  # second token is not a digit -> first token only
    ("   ", "   "),  # falsy after strip -> returned unchanged
    ("", ""),
]


@pytest.mark.parametrize(("name", "expected"), _SPECIES_CASES, ids=lambda v: repr(v))
def test_parse_cloudy_species(name, expected):
    """The species parser, including both fallbacks and the blank-name path."""
    assert _parse_cloudy_species(name) == expected


#: (name, species, is_balmer). H1 lines match either by keyword or, for
#: CLOUDY-format names, by falling in the Balmer [3646, 6563] or Lyman
#: [912, 1216] A window. A non-H1 species is never Balmer.
_BALMER_CASES = [
    ("Halpha", "H1", True),
    ("Hbeta", "H1", True),
    ("Hgamma", "H1", True),
    ("Hdelta", "H1", True),
    ("H  1 4861.33A", "H1", True),  # CLOUDY Hbeta, inside the Balmer window
    ("H  1 1215.67A", "H1", True),  # CLOUDY Lya, inside the Lyman window
    ("H  1 18751.0A", "H1", False),  # Paschen, outside both windows
    ("H  1 BADWA", "H1", False),  # malformed wavelength falls through
    ("Lya", "H1", False),  # no keyword match, and too few tokens to parse
    ("OIII_5007", "O3", False),
    ("NII_6584", "N2", False),
]


@pytest.mark.parametrize(("name", "species", "expected"), _BALMER_CASES)
def test_is_balmer_line(name, species, expected):
    """Balmer classification by keyword, by CLOUDY wavelength, and by species."""
    assert _is_balmer_line(name, species) is expected


#: (name, species, is_broad_candidate). Broad by species (H1 and the permitted
#: high-ionization ions) or by a keyword in the name, which is what lets an
#: unknown species still be flagged.
_BROAD_CASES = [
    ("Halpha", "H1", True),
    ("CIV_1549", "C4", True),
    ("MgII_2796", "Mg2", True),
    ("CIII_1908", "C3", True),
    ("OIII_5007", "O3", False),
    ("NII_6584", "N2", False),
    ("SII_6717", "S2", False),
    ("Halpha_broad", "XX", True),  # keyword wins over an unknown species
    ("civ_line", "XX", True),
]


@pytest.mark.parametrize(("name", "species", "expected"), _BROAD_CASES)
def test_is_broad_candidate(name, species, expected):
    """Broad-line candidacy by species and by name keyword."""
    assert _is_broad_candidate(name, species) is expected


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
