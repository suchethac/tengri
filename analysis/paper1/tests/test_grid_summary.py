# SPDX-License-Identifier: BSD-3-Clause
"""Grid-wide numbers must not be computed from part of a grid.

Section 7's nine TBDs are aggregates over twenty galaxies by six
configurations. Row III finished first, and a summary that happily reports
"19 of 20 adopted" from it would be arithmetic over a sixth of the grid
wearing the grid's name -- and unlike a blank, that is quotable.

The second refusal is subtler. ``adoption_pass`` means ``n_divergent == 0 and
rhat_max < 1.01`` on the paper's pin and additionally ``ess_min >= 100`` on the
grid session's driver, so cells written either side of 03ffff576 carry one key
holding two predicates. Summing them without saying which is which produces an
adoption count for a bar that no cell was actually judged against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.grid_summary import (
    EXPECTED_CONFIGS,
    EXPECTED_GALAXIES,
    Missing,
    load_cells,
    per_cell,
    summarize,
)

pytestmark = pytest.mark.unit

CONFIGS = ["I", "II", "III", "IV", "V", "VI"]


def _cell(gal, config, *, ess=250.0, wall=1800.0, adopted=True, rhat=1.004, three_leg=False):
    cell = {
        "gal_id": gal,
        "config": config,
        "adoption_pass": adopted,
        "ess_min": ess,
        "rhat_max": rhat,
        "divergences": 0,
        "wall_time_s": wall,
        "retune_attempt": 1,
    }
    if three_leg:
        cell["priors"] = {"noted": "post-03ffff576 schema"}
    return cell


def _rows(cells):
    return [per_cell(f"{c['gal_id']}_{c['config']}", c) for c in cells]


def _full_grid(**kw):
    return [_cell(g, c, **kw) for g in range(1, EXPECTED_GALAXIES + 1) for c in CONFIGS]


def test_one_row_of_six_withholds_every_aggregate():
    """The defect. Row III alone cannot answer a grid-wide question."""
    rows = _rows([_cell(g, "III") for g in range(1, EXPECTED_GALAXIES + 1)])

    summary = summarize(rows)

    assert summary["aggregates"] is None
    assert "1 of 6 configurations" in summary["aggregates_withheld_because"]
    # The per-configuration rows are still useful and still reported.
    assert summary["per_configuration"]["III"]["cells"] == EXPECTED_GALAXIES


def test_a_complete_grid_reports_them():
    """Otherwise the refusal would be a permanent block on a finished grid."""
    summary = summarize(_rows(_full_grid()))

    assert summary["grid_complete"] is True
    agg = summary["aggregates"]
    assert agg["n_adopted"] == EXPECTED_GALAXIES * EXPECTED_CONFIGS
    assert agg["ess_min"] == 250.0
    assert agg["mixes_worst"] in CONFIGS


def test_missing_galaxies_withhold_even_with_every_configuration():
    """Six configurations on three galaxies is not the grid either."""
    rows = _rows([_cell(g, c) for g in (1, 2, 3) for c in CONFIGS])

    summary = summarize(rows)

    assert summary["aggregates"] is None
    assert "3 of 20 galaxies" in summary["aggregates_withheld_because"]


def test_the_two_adoption_bars_are_reported_apart():
    """One key, two predicates. The census says which cells are which."""
    cells = [_cell(g, c) for g in range(1, EXPECTED_GALAXIES + 1) for c in CONFIGS]
    for cell in cells[:4]:
        cell["priors"] = {"noted": "post-03ffff576"}

    summary = summarize(_rows(cells))

    split = summary["adoption_bar_split"]
    assert split["three_leg_cells"] == 4
    assert split["two_leg_cells"] == len(cells) - 4
    assert "03ffff576" in split["note"]


def test_a_missing_ess_refuses_rather_than_defaulting():
    cell = _cell(1, "III")
    del cell["ess_min"]
    with pytest.raises(Missing) as excinfo:
        per_cell("1_III", cell)
    assert "ess_min" in str(excinfo.value)


def test_a_null_diagnostic_is_as_absent_as_a_missing_one():
    """``"rhat_max": null`` is an unrecorded value, not a passing one."""
    cell = _cell(1, "III")
    cell["rhat_max"] = None
    with pytest.raises(Missing):
        per_cell("1_III", cell)


def test_a_zero_ess_refuses_because_seconds_per_sample_is_undefined():
    """A frozen chain would otherwise divide by zero or report infinity."""
    with pytest.raises(Missing) as excinfo:
        per_cell("1_III", _cell(1, "III", ess=0.0))
    assert "undefined" in str(excinfo.value)


def test_seconds_per_effective_sample_is_wall_over_ess():
    row = per_cell("1_III", _cell(1, "III", ess=200.0, wall=1000.0))
    assert row["s_per_ess"] == pytest.approx(5.0)


def test_an_empty_results_directory_is_refused(tmp_path):
    """Absence of cells is not a grid with nothing to say."""
    with pytest.raises(SystemExit) as excinfo:
        load_cells(tmp_path)
    assert "has not run" in str(excinfo.value)


def test_a_missing_results_directory_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        load_cells(tmp_path / "nope")


def test_worst_mixing_configuration_is_the_one_with_the_lowest_adopted_ess():
    cells = _full_grid()
    for cell in cells:
        if cell["config"] == "V":
            cell["ess_min"] = 110.0

    summary = summarize(_rows(cells))

    assert summary["aggregates"]["mixes_worst"] == "V"
    assert summary["per_configuration"]["V"]["ess_min_worst"] == 110.0


def test_a_refused_cell_does_not_set_the_worst_adopted_ess(tmp_path):
    """An unadopted cell's ESS is not the grid's minimum effective sample.

    Section 7 quotes the minimum over cells the paper *draws*. Folding a
    refused cell into it would report a number from a posterior the figure
    never shows.
    """
    cells = _full_grid()
    cells[0]["adoption_pass"] = False
    cells[0]["ess_min"] = 4.0

    summary = summarize(_rows(cells))

    assert summary["aggregates"]["ess_min"] == 250.0
    assert summary["aggregates"]["n_adopted"] == len(cells) - 1


def test_the_summary_round_trips_as_json():
    """It is written to disk and read by whoever fills the section."""
    summary = summarize(_rows(_full_grid()))
    assert json.loads(json.dumps(summary))["aggregates"]["n_adopted"] == 120
