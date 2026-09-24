# SPDX-License-Identifier: BSD-3-Clause
"""``grid_census.py`` must not let a partial grid answer for the whole one.

Section 7's seventeen open entries are statistics over a finished twenty by
six grid. The census exists so they are computed rather than transcribed, and
the two things it has to get right are the two the hand-fill kept getting
wrong: a field a cell does not carry is not a zero, and nineteen cells of one
configuration are not a statement about six.

Both are load-bearing rather than decorative. The paper has already described
a machine nobody measured and an adoption bar with one fewer leg than the
grid applies; a census that quietly defaulted a missing ``ess_min`` to zero,
or that printed row III's spread as the grid's, would put the same class of
claim back in by a shorter route.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parents[1]
CENSUS = ANALYSIS_DIR / "grid_census.py"
SELECTION_20 = ANALYSIS_DIR / "results" / "selected_galaxies_20.json"

sys.path.insert(0, str(ANALYSIS_DIR))

pytestmark = pytest.mark.contract

PARTIAL_BANNER = "PARTIAL GRID"


def _cell(gal_id: int, config: str = "III", **overrides) -> dict:
    """A cell that clears the adoption bar unless an override breaks it."""
    cell = {
        "gal_id": gal_id,
        "config": config,
        "adoption_pass": True,
        "divergences": 0,
        "rhat_max": 1.002,
        "ess_min": 300.0,
        "wall_time_s": 1000.0,
        "retune_attempt": 1,
        "profile_mass": True,
        "load_at_start": {"n_cpus": 24, "load_avg_1m": 30.0, "n_concurrent_fits": 4},
        "attempts": [{"retune_attempt": 1}],
    }
    cell.update(overrides)
    return cell


def _write(results_dir: Path, cells: dict[str, dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, cell in cells.items():
        (results_dir / f"{name}.json").write_text(json.dumps(cell))
        np.savez(results_dir / f"{name}.npz", dust_tau_diff=np.zeros(4))


def _run(results_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CENSUS), "--results-dir", str(results_dir), *extra],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "JAX_PLATFORMS": "cpu",
        },
        timeout=900,
    )


@pytest.fixture(autouse=True)
def _require_selection():
    if not SELECTION_20.is_file():
        pytest.skip(f"locked selection missing: {SELECTION_20.name}")


def test_a_partial_grid_exits_nonzero_and_says_so(tmp_path):
    """The guard against row III's numbers being read as the grid's."""
    results = tmp_path / "fits"
    _write(results, {f"{gid}_III": _cell(gid) for gid in (79, 4171, 13097)})

    result = _run(results)

    assert result.returncode != 0, (
        "a three-cell grid was reported as if it answered for one hundred and twenty"
    )
    assert PARTIAL_BANNER in result.stdout, (
        f"no partial-grid banner in the output.\nstdout:\n{result.stdout[-1200:]}"
    )
    assert "3 of 120" in result.stdout, (
        f"the count was not stated.\nstdout:\n{result.stdout[-1200:]}"
    )


def test_allow_partial_is_the_only_way_to_a_zero_exit(tmp_path):
    """Watching a run is legitimate; filling the paper from one is not.

    The flag must still print the banner -- an override that also silenced the
    warning would make the two cases indistinguishable in a log.
    """
    results = tmp_path / "fits"
    _write(results, {f"{gid}_III": _cell(gid) for gid in (79, 4171)})

    result = _run(results, "--allow-partial")

    assert result.returncode == 0, f"--allow-partial did not succeed: {result.stderr[-600:]}"
    assert PARTIAL_BANNER in result.stdout, "--allow-partial silenced the banner"


def test_a_missing_field_is_reported_absent_rather_than_zero():
    """`meta.get(x) or 0` is how a missing diagnostic became a clean one."""
    from grid_census import coverage

    cells = {
        "a": {"ess_min": 300.0},
        "b": {"ess_min": None},
        "c": {},
    }
    present, missing = coverage(cells, "ess_min")

    assert present == [300.0], f"absent values leaked into the sample: {present}"
    assert missing == 2, (
        f"counted {missing} cells as lacking ess_min, expected 2. A census that "
        "defaults the absent ones reports a minimum of zero over a grid whose "
        "cells simply did not record it."
    )


def test_a_cell_with_no_ess_min_is_disqualified_not_summarized(tmp_path):
    """End to end, the rule turns out to be stronger than "reported absent".

    Configuration III is judged on the relaxed bar, which refuses a cell that
    records no ``ess_min`` outright -- absent means unverified, so the cell
    never reaches the adopted set and cannot pull its minimum anywhere. This
    pins that, not the weaker property that the value is merely excluded from
    the average.
    """
    results = tmp_path / "fits"
    cells = {"79_III": _cell(79, ess_min=300.0), "4171_III": _cell(4171, ess_min=450.0)}
    cells["13097_III"] = _cell(13097, ess_min=None)
    _write(results, cells)

    result = _run(results, "--allow-partial")

    # Read the reported figure rather than substring-matching it: "0.0" is
    # inside "300.0", which is how the first version of this test failed on a
    # census that was behaving correctly.
    line = result.stdout.split("adopted ess_min")[-1].splitlines()[0]
    reported = line.split(":", 1)[-1].split("(")[0].strip()
    assert reported == "300.0 to 450.0", (
        f"reported {reported!r}; the two recorded values are 300.0 and 450.0, so "
        "anything lower means the cell that records no ess_min was counted as one "
        f"that recorded zero.\nstdout:\n{result.stdout[-1200:]}"
    )
    assert "13097_III" in result.stdout.split("not adopted")[-1].splitlines()[0], (
        "the cell recording no ess_min was not named among the refusals; a "
        "missing diagnostic has to disqualify a cell, not be summarized over.\n"
        f"stdout:\n{result.stdout[-1200:]}"
    )
    assert "adopted              : 2 of 3" in result.stdout, (
        f"expected two of three cells adopted.\nstdout:\n{result.stdout[-1200:]}"
    )


def test_band_residuals_are_in_the_units_the_fit_was_scored_in(tmp_path):
    """(model - observed) / sigma, with the sigma the likelihood saw.

    fit_one hands save_fit_outputs the floored sigma, so a residual computed
    from the cell is already in the units the objective used. Reconstructing
    it from raw catalog errors would overstate every misfit.
    """
    from grid_census import band_residuals

    results = tmp_path / "fits"
    results.mkdir()
    np.savez(
        results / "79_III.npz",
        obs_fnu=np.array([1.0, 2.0, 4.0]),
        obs_sigma=np.array([0.1, 0.5, 1.0]),
        model_photometry_median=np.array([1.2, 2.0, 2.0]),
        filter_names=np.array(["a", "b", "c"], dtype=object),
    )

    residuals, bands = band_residuals(results, "79_III")

    assert bands == ["a", "b", "c"]
    np.testing.assert_allclose(residuals, [2.0, 0.0, -2.0])


def test_a_cell_with_no_arrays_is_counted_not_skipped_silently(tmp_path):
    """A missing npz must not quietly shrink the sample the census reports."""
    from grid_census import band_residuals

    results = tmp_path / "fits"
    results.mkdir()
    assert band_residuals(results, "79_III") is None, (
        "a cell with no photometry arrays returned residuals from nowhere"
    )


def test_the_census_reports_fit_quality_separately_from_the_bar(tmp_path):
    """The bar is sampler convergence; chi2 is whether the model fits."""
    results = tmp_path / "fits"
    results.mkdir()
    for gid, model in ((79, 1.0), (4171, 5.0)):
        (results / f"{gid}_III.json").write_text(json.dumps(_cell(gid)))
        np.savez(
            results / f"{gid}_III.npz",
            obs_fnu=np.array([1.0, 1.0, 1.0]),
            obs_sigma=np.array([0.1, 0.1, 0.1]),
            model_photometry_median=np.array([model, model, model]),
            filter_names=np.array(["a", "b", "c"], dtype=object),
        )

    result = _run(results, "--allow-partial")

    assert "chi2 per band" in result.stdout, (
        f"the census did not report fit quality.\nstdout:\n{result.stdout[-1200:]}"
    )
    # Galaxy 4171's model is 40 sigma off in every band while its cell records
    # adoption_pass True: exactly the case the bar cannot see.
    assert "cells above 2       : 1 of 2" in result.stdout, (
        "a cell that clears the adoption bar while missing every band by forty "
        f"sigma was not counted as a poor fit.\nstdout:\n{result.stdout[-1200:]}"
    )


def test_one_bad_band_is_distinguished_from_a_model_that_misses_broadly(tmp_path):
    """The two look identical in chi2 and want opposite responses.

    A cell whose misfit is one bad photometric point collapses to a good fit
    when that point is dropped; a cell whose model misses everywhere does not.
    Reporting only the headline chi2 leaves those indistinguishable, which is
    how a handful of bad IRAC deblends could be read as a broken model, or a
    broken model excused as bad photometry.
    """
    results = tmp_path / "fits"
    results.mkdir()

    # 79: four bands perfect, one 10 sigma out. 4171: every band 3 sigma out.
    (results / "79_III.json").write_text(json.dumps(_cell(79)))
    np.savez(
        results / "79_III.npz",
        obs_fnu=np.ones(5),
        obs_sigma=np.full(5, 0.1),
        model_photometry_median=np.array([1.0, 1.0, 1.0, 1.0, 2.0]),
        filter_names=np.array(list("abcde"), dtype=object),
    )
    (results / "4171_III.json").write_text(json.dumps(_cell(4171)))
    np.savez(
        results / "4171_III.npz",
        obs_fnu=np.ones(5),
        obs_sigma=np.full(5, 0.1),
        model_photometry_median=np.full(5, 1.3),
        filter_names=np.array(list("abcde"), dtype=object),
    )

    result = _run(results, "--allow-partial")

    assert "worst band dropped" in result.stdout, (
        f"the census reported no trimmed chi2.\nstdout:\n{result.stdout[-1200:]}"
    )
    line = result.stdout.split("worst band dropped")[-1].splitlines()[0]
    # 79 trims to exactly 0; 4171 stays at 9. A range that starts anywhere but
    # zero means the outlier-driven cell was not actually trimmed.
    assert "0.00 to 9.00" in line, (
        "the trimmed range should span the outlier-driven cell (0.00) and the "
        f"broadly-wrong one (9.00); got {line.strip()!r}"
    )


def test_cells_outside_the_requested_configurations_do_not_fill_the_quota(tmp_path, capsys):
    """A foreign-configuration cell must not be counted toward a subset view.

    ``report`` decides completeness with ``have == total``, and both sides can
    be wrong in opposite directions: a cell from a configuration the caller did
    not ask for inflates ``have``, while the shorter ``config_keys`` shrinks
    ``total``. The two errors cancel, and the census prints COMPLETE for a grid
    with a hole in it -- the reassuring answer, which is the dangerous one.

    Here two galaxies and two configurations want four cells. Three are present
    and ``2_II`` is missing, so the honest answer is partial; a single stray
    Configuration VI cell brings the unfiltered count to four.
    """
    import grid_census as gc

    cells = {
        "1_I": _cell(1, "I"),
        "2_I": _cell(2, "I"),
        "1_II": _cell(1, "II"),
        "1_VI": _cell(1, "VI"),  # not asked for; must not fill 2_II's slot
    }
    complete = gc.report(cells, [1, 2], ["I", "II"], tmp_path)
    out = capsys.readouterr().out

    assert complete is False, "a grid missing 2_II was reported complete"
    assert "3 of 4" in out, f"foreign cell counted toward the quota:\n{out}"
    assert "VI" not in out.split("configurations seen")[1].split("\n")[0]
