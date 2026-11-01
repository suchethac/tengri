# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``LineList.default_optical()`` carries H8 to H14, He I 3889 and [Ne III] 3968.

Hydrogen wavelengths are the NIST ASD H I Ritz values (energy levels fetched 2026-10-31
from https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=H+I&units=0&format=2&output=0&page_size=15&multiplet_ordered=0&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&g_out=on&submit=Retrieve+Data;
vacuum Ritz wavelength 1e8 / (E_n - E_2)) tabulated in
``tengri.components.nebular._balmer_series``; the catalog agrees with them to 0.02 A and with
Cue's own H8 to H10 (Cloudy air labels through ``air_to_vac``) to 0.04 A. He I 3889 and
[Ne III] 3968 are Cue's catalog lines (Cloudy labels, vacuum). The doublet constraint matrix
gains one row and one column per added line and no new constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.nebular._balmer_series import (
    BALMER_CASE_B_RATIOS_TO_HBETA,
    BALMER_RITZ_VACUUM_AA_N3_TO_N10,
)
from tengri.observation.line_list import LineList

pytestmark = pytest.mark.contract

_ADDED = (
    "H14",
    "H13",
    "H12",
    "H11",
    "H10",
    "H9",
    "HeI_3889",
    "H8",
    "NeIII_3968",
)
_RITZ = {f"H{n}": lam for n, lam in BALMER_RITZ_VACUUM_AA_N3_TO_N10}
_RITZ.update({f"H{n}": lam for n, lam, _ in BALMER_CASE_B_RATIOS_TO_HBETA})
_CUE_VACUUM = {"HeI_3889": 3889.742, "NeIII_3968": 3968.593}
_PRE_EXISTING_LINES = 39
_PRE_EXISTING_DOUBLETS = 5


def _wave(cat, name):
    return float(np.asarray(cat.wavelengths)[cat.names.index(name)])


def test_added_lines_exist_with_vacuum_wavelengths():
    """Each added line is in the catalog at its NIST (hydrogen) or Cue-catalog (He, Ne) value."""
    cat = LineList.default_optical()
    for name in _ADDED:
        assert name in cat.names, name
    for name in _ADDED:
        expect = _CUE_VACUUM.get(name, _RITZ.get(name))
        assert abs(_wave(cat, name) - expect) < 0.02, name


def test_hydrogen_series_increases_with_decreasing_n_and_catalog_is_sorted():
    """H14 -> H8 are strictly increasing in wavelength; the whole catalog stays sorted."""
    cat = LineList.default_optical()
    series = [_wave(cat, f"H{n}") for n in range(14, 7, -1)]
    assert np.all(np.diff(series) > 0.0)
    waves = np.asarray(cat.wavelengths)
    assert np.all(np.diff(waves) > 0.0)
    assert cat.names.index("H8") < cat.names.index("Hepsilon") < cat.names.index("Hdelta")


def test_added_lines_are_balmer_or_their_own_species():
    """The hydrogen entries are flagged Balmer with species H1; He I and [Ne III] are not."""
    cat = LineList.default_optical()
    for n in range(8, 15):
        i = cat.names.index(f"H{n}")
        assert cat.species[i] == "H1" and cat.is_balmer[i]
    assert cat.species[cat.names.index("HeI_3889")] == "He1"
    assert cat.species[cat.names.index("NeIII_3968")] == "Ne3"
    assert not cat.is_balmer[cat.names.index("NeIII_3968")]


def test_constraint_matrix_grows_by_the_added_lines_with_no_new_constraint():
    """9 lines added: n_lines and n_independent both grow by 9; the doublet count is unchanged."""
    cat = LineList.default_optical()
    assert cat.n_lines == _PRE_EXISTING_LINES + len(_ADDED)
    assert len(cat.doublets) == _PRE_EXISTING_DOUBLETS
    mat = np.asarray(cat.build_constraint_matrix())
    assert mat.shape == (cat.n_lines, cat.n_lines - _PRE_EXISTING_DOUBLETS)
    for name in _ADDED:
        row = mat[cat.names.index(name)]
        assert row.sum() == 1.0 and np.count_nonzero(row) == 1


def test_default_13_is_unchanged():
    """The legacy 13-line catalog keeps its lines."""
    assert LineList.default_13().n_lines == 13
