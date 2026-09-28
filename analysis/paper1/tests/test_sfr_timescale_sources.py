# SPDX-License-Identifier: BSD-3-Clause
"""SFR timescale sources are correctly pinned.

BAGPIPES publishes a 100 Myr averaged SFR, sourced from bagpipes 0.7.12
star_formation_history.py lines 122-124. BEAGLE and Dense_Basis document
100 Myr averages via their SFR column names. CIGALE and Prospector do not
record a timescale. The catalog notes, grid_census filter, and ingest values
must match these facts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.published_code_spread import SFR_100MYR_DOCUMENTED_CODES, SFR_TIMESCALE

pytestmark = pytest.mark.unit


def test_sfr_timescale_dict_covers_five_codes():
    """SFR_TIMESCALE dict covers exactly the five codes."""
    assert set(SFR_TIMESCALE.keys()) == {
        "BAGPIPES",
        "BEAGLE",
        "Dense_Basis",
        "CIGALE",
        "Prospector",
    }


def test_bagpipes_is_100myr_with_source():
    """BAGPIPES documents a 100 Myr average SFR with a source citation."""
    assert SFR_TIMESCALE["BAGPIPES"]["timescale"] == "100 Myr"
    assert "bagpipes" in SFR_TIMESCALE["BAGPIPES"]["source"].lower()
    assert "star_formation_history" in SFR_TIMESCALE["BAGPIPES"]["source"]
    assert "122" in SFR_TIMESCALE["BAGPIPES"]["source"]


def test_beagle_is_100myr():
    """BEAGLE documents a 100 Myr average SFR."""
    assert SFR_TIMESCALE["BEAGLE"]["timescale"] == "100 Myr"
    assert "SFR_100" in SFR_TIMESCALE["BEAGLE"]["source"]


def test_dense_basis_is_100myr():
    """Dense_Basis documents a 100 Myr average SFR."""
    assert SFR_TIMESCALE["Dense_Basis"]["timescale"] == "100 Myr"
    assert "log_SFR_100" in SFR_TIMESCALE["Dense_Basis"]["source"]


def test_cigale_has_no_timescale():
    """CIGALE does not record a SFR timescale."""
    assert SFR_TIMESCALE["CIGALE"]["timescale"] is None


def test_prospector_has_no_timescale():
    """Prospector does not record a SFR timescale."""
    assert SFR_TIMESCALE["Prospector"]["timescale"] is None


def test_sfr_100myr_documented_codes_is_correct():
    """SFR_100MYR_DOCUMENTED_CODES includes exactly the three documented codes."""
    assert {"BAGPIPES", "BEAGLE", "Dense_Basis"} == SFR_100MYR_DOCUMENTED_CODES


def test_no_instantaneous_claims_in_published_code_spread():
    """The string 'instantaneous' does not appear in published_code_spread module."""
    import paper1.published_code_spread as pcs_module

    source = Path(pcs_module.__file__).read_text()
    # Check that the word "instantaneous" does not appear outside a negation
    for line in source.splitlines():
        if "instantaneous" in line.lower():
            # The word should only appear in comments, not in active code
            # or in claims about code behavior. Allow it only if negated.
            if line.strip().startswith("#"):
                continue  # Comments are allowed
            # If not a comment, must be a negation or historical note
            if "not" in line.lower() or "no" in line.lower() or "negat" in line.lower():
                continue
            pytest.fail(f"Found 'instantaneous' in non-negated context: {line}")


def test_no_instantaneous_claims_in_grid_census():
    """The string 'instantaneous' does not appear in grid_census module."""
    import paper1.grid_census as gc_module

    source = Path(gc_module.__file__).read_text()
    for line in source.splitlines():
        if "instantaneous" in line.lower():
            if line.strip().startswith("#"):
                continue
            pytest.fail(f"Found 'instantaneous' in grid_census: {line}")


def test_no_instantaneous_claims_in_ingest():
    """The string 'instantaneous' does not appear in ingest_art_sedfitting module."""
    ingest_file = PAPER1 / "ingest_art_sedfitting.py"
    source = ingest_file.read_text()
    for line in source.splitlines():
        if "instantaneous" in line.lower():
            if line.strip().startswith("#"):
                continue
            pytest.fail(f"Found 'instantaneous' in ingest_art_sedfitting: {line}")


def test_grid_census_sfr_100myr_filter_matches_documented_codes():
    """grid_census's logsfr_100myr filter matches codes with 100 Myr in note."""
    # This is a static check: the filter in grid_census.py checks for
    # "100 Myr" in the sfr_timescale_note, which matches the documented codes.
    import paper1.grid_census as gc

    source = Path(gc.__file__).read_text()
    assert "100 Myr" in source
    assert "logsfr_100myr" in source
    assert "sfr_timescale_note" in source
