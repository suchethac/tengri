# SPDX-License-Identifier: BSD-3-Clause
"""A partial grid must not render as the whole sample.

``fig09_sample_level.py`` chose its provenance stamp on directory identity:
PROVISIONAL when ``--results-dir`` was not the canonical path. "Where did this
come from" is not "is all of it here", and the two separated the moment the
production grid wrote its first cells into the canonical directory. Five cells
of a hundred and twenty then rendered with **no stamp**, under a caption
describing twenty galaxies by six configurations.

The regression test is the canonical-directory case: a partial grid sitting at
exactly the blessed path, which is the one arrangement the old check could not
see.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _grid_completeness import (
    completeness_note,
    load_expected_galaxy_ids,
    missing_cells,
)

pytestmark = pytest.mark.contract

CONFIGS_SIX = ["I", "II", "III", "IV", "V", "VI"]
SELECTION_20 = Path(__file__).resolve().parents[1] / "results" / "selected_galaxies_20.json"


def _all_cells(galaxy_ids, configs):
    return [(gal, cfg) for gal in galaxy_ids for cfg in configs]


def test_a_complete_grid_gets_no_note():
    """The check must be able to pass, or its failures mean nothing."""
    ids = [1, 2, 3]
    assert completeness_note(_all_cells(ids, CONFIGS_SIX), ids, CONFIGS_SIX) is None


def test_a_partial_grid_is_named_with_its_shortfall():
    ids = [1, 2, 3]
    present = [(1, "I"), (1, "II")]
    note = completeness_note(present, ids, CONFIGS_SIX)
    assert note is not None
    assert "2 of 18" in note
    assert "1_III" in note


def test_missing_cells_lists_exactly_what_is_absent():
    ids = [7, 9]
    present = [(7, "I"), (9, "II")]
    absent = missing_cells(present, ids, ["I", "II"])
    assert absent == ["7_II", "9_I"]


def test_the_five_of_one_hundred_and_twenty_case():
    """The exact arrangement that rendered unstamped."""
    ids = list(range(1, 21))
    present = [(13097, "III")][:0] + [(ids[i], "III") for i in range(4)] + [(ids[4], "V")]
    note = completeness_note(present, ids, CONFIGS_SIX)
    assert note is not None and "5 of 120" in note


def test_the_expected_sample_is_read_from_the_committed_selection():
    """Derived, not restated: a re-locked sample must not leave a stale 20."""
    ids = load_expected_galaxy_ids(SELECTION_20)
    assert len(ids) == 20
    assert len(set(ids)) == 20, "the locked selection repeats a galaxy"
    assert all(isinstance(i, int) for i in ids)


def test_an_empty_selection_raises_rather_than_declaring_everything_complete(tmp_path):
    """Zero expected cells would make every render trivially complete."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"selected_galaxies": []}))
    with pytest.raises(ValueError):
        load_expected_galaxy_ids(path)


# --- the wiring, not just the logic -----------------------------------------


def _write_cell(results_dir: Path, gal_id: int, config: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{gal_id}_{config}.json").write_text(
        json.dumps({"adoption_pass": True, "n_samples": 1000})
    )
    rng = np.random.default_rng(gal_id)
    np.savez(
        results_dir / f"{gal_id}_{config}.npz",
        stellar_mass=10 ** rng.normal(10.5, 0.1, 64),
        sfr_100myr=10 ** rng.normal(0.5, 0.1, 64),
    )


def test_a_partial_grid_in_the_canonical_directory_is_reported(tmp_path, monkeypatch, capsys):
    """The regression. The old check passed this arrangement silently.

    The shortfall used to be drawn onto the figure. It is not: a developmental
    note has no place on a published panel, so the figure stays clean and the
    shortfall goes to stderr and the sidecar. What must not change is that a
    two-cell grid sitting at the canonical path is *reported* rather than
    rendering as the full twenty-by-six sample.
    """
    import fig09_sample_level as fig09

    results = tmp_path / "fits"
    for gal, cfg in [(79, "III"), (267, "III")]:
        _write_cell(results, gal, cfg)

    # Make the partial directory *be* the canonical one, which is precisely
    # what the directory-identity check could not distinguish.
    monkeypatch.setattr(fig09, "CANONICAL_RESULTS", results)

    def fake_build_figure(cells):
        return fig09.plt.figure(), {
            "n_cells": len(cells),
            "n_galaxies": 2,
            "n_adopted": len(cells),
            "representative_galaxy": 79,
            "sample_sigma_log_mstar": 0.1,
            "sample_sigma_log_sfr": 0.2,
        }

    monkeypatch.setattr(fig09, "build_figure", fake_build_figure)
    rc = fig09.main(["--results-dir", str(results), "--out", str(tmp_path / "o.pdf")])

    assert rc == 0
    err = capsys.readouterr().err
    assert "INCOMPLETE" in err, (
        "a two-cell grid at the canonical path was rendered with no shortfall "
        f"reported anywhere; the figure would pass as the full sample.\nstderr:\n{err}"
    )


def test_present_on_disk_requires_both_files(tmp_path):
    """A JSON with no posterior beside it is not a rendered cell.

    fig09's loader already skips such a cell, so counting it as present would
    make the completeness note disagree with what actually got drawn.
    """
    from _grid_completeness import present_on_disk

    _write_cell(tmp_path, 13097, "I")
    (tmp_path / "13097_II.json").write_text(json.dumps({"adoption_pass": True}))  # no NPZ

    found = present_on_disk(tmp_path, [13097], ["I", "II"])
    assert found == [(13097, "I")]


def test_present_on_disk_ignores_cells_outside_the_declared_sample(tmp_path):
    """A stray galaxy must not fill a hole in the sample that is declared."""
    from _grid_completeness import present_on_disk

    _write_cell(tmp_path, 99999, "I")
    assert present_on_disk(tmp_path, [13097], ["I"]) == []


def test_present_on_disk_feeds_a_shortfall_note(tmp_path):
    """The fig05 arrangement: three galaxies, six configurations, few cells."""
    from _grid_completeness import present_on_disk

    ids = [13097, 15336, 16049]
    _write_cell(tmp_path, 13097, "I")
    _write_cell(tmp_path, 15336, "II")
    note = completeness_note(present_on_disk(tmp_path, ids, CONFIGS_SIX), ids, CONFIGS_SIX)
    assert note is not None and "2 of 18" in note


def test_present_on_disk_sees_a_whole_grid(tmp_path):
    """Non-vacuity: the scan must be able to report completeness."""
    from _grid_completeness import present_on_disk

    ids = [1, 2]
    for gal in ids:
        for cfg in ["I", "II"]:
            _write_cell(tmp_path, gal, cfg)
    found = present_on_disk(tmp_path, ids, ["I", "II"])
    assert len(found) == 4
    assert completeness_note(found, ids, ["I", "II"]) is None


# --- fig09 must judge cells with the shared rule, not its own copy ----------


def _cell_with(results_dir: Path, gal_id: int, config: str, meta: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{gal_id}_{config}.json").write_text(json.dumps(meta))
    rng = np.random.default_rng(gal_id)
    np.savez(
        results_dir / f"{gal_id}_{config}.npz",
        stellar_mass=10 ** rng.normal(10.5, 0.1, 64),
        sfr_100myr=10 ** rng.normal(0.5, 0.1, 64),
    )


def test_fig09_judges_with_the_shared_rule_not_the_raw_flag(tmp_path):
    """A Configuration III cell the relaxed bar accepts must be drawn adopted.

    fig09 read meta["adoption_pass"] directly while fig05 and fig06 judged
    through _adoption.is_adopted. That made the sample-level figure -- the one
    an adoption rate is read off -- the only consumer with its own copy of the
    criterion. This cell separates the two: the raw flag says no, the relaxed
    bar for III says yes.
    """
    import fig09_sample_level as fig09

    results = tmp_path / "fits"
    _cell_with(
        results,
        13097,
        "III",
        {
            "adoption_pass": False,
            "rhat_max": 1.004,
            "divergences": 0,
            "ess_min": 300.0,
            "n_samples": 300,
            "n_chains": 4,
        },
    )
    cells, _ = fig09.load_cells(results)
    assert len(cells) == 1
    assert cells[0].adopted, "fig09 fell back to the raw flag instead of the shared rule"


def test_fig09_marks_a_cell_adopted_on_too_few_effective_samples(tmp_path):
    """The detector must reach the figure, not just exist."""
    import fig09_sample_level as fig09

    results = tmp_path / "fits"
    _cell_with(
        results,
        14099,
        "V",
        {
            "adoption_pass": True,
            "rhat_max": 1.0089,
            "divergences": 0,
            "ess_min": 2.8,
            "n_samples": 300,
            "n_chains": 4,
        },
    )
    cells, _ = fig09.load_cells(results)
    assert cells[0].adopted
    assert cells[0].low_ess is not None
    assert "2.8" in cells[0].low_ess


def test_fig09_leaves_an_honest_cell_unmarked(tmp_path):
    """Non-vacuity: the mark must not appear on every cell."""
    import fig09_sample_level as fig09

    results = tmp_path / "fits"
    _cell_with(
        results,
        9884,
        "V",
        {
            "adoption_pass": True,
            "rhat_max": 1.0037,
            "divergences": 0,
            "ess_min": 380.8,
            "n_samples": 300,
            "n_chains": 4,
        },
    )
    cells, _ = fig09.load_cells(results)
    assert cells[0].adopted and cells[0].low_ess is None
