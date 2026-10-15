# SPDX-License-Identifier: BSD-3-Clause
"""``fig06_code_overlay.py`` must not draw a comparison with no tengri arm.

The figure's claim is tengri's posteriors beside the published per-code values.
Pointed at a directory holding no finished cells it logged every one of them as
"not ready" at INFO level, drew the published points alone, saved a PDF under
the paper's filename and exited 0.

The provenance audit in front of it could not catch this. It asks whether the
cells that *are* present hold the model ``configs.py`` declares, and a directory
with no cells has no cell that disagrees -- it passed, noting only that nothing
was verified. Absence read as agreement.

These tests pin the two halves of the fix, and the second is the one that keeps
the first honest: refusing *every* incomplete grid would also pass test one
while making the figure unbuildable until the last of a hundred and twenty cells
landed. A partial grid draws, stamped; an empty one refuses.
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
FIG06 = ANALYSIS_DIR / "fig06_code_overlay.py"

sys.path.insert(0, str(ANALYSIS_DIR))

from _cell_provenance import SFH_PREFIX_BY_TYPE
from _grid_completeness import load_expected_galaxy_ids
from config_metadata import CONFIGS
from fig06_code_overlay import PANEL_GALAXY_IDS, SELECTION_20

pytestmark = pytest.mark.contract

EMPTY_REFUSAL = "no finished cells"


def _run_fig06(results_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    """Invoke the figure exactly as a reader would, and never raise on failure."""
    return subprocess.run(
        [
            sys.executable,
            str(FIG06),
            "--results-dir",
            str(results_dir),
            "--out-dir",
            str(out_dir),
        ],
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


def _stub_cell(results_dir: Path, gal_id: int, config: str) -> None:
    """The smallest cell the provenance audit will accept for `config`.

    The SFH prefix is read off the declared configuration rather than written
    in, so a configuration that changes its SFH family cannot leave this test
    quietly fabricating a cell the audit would reject.
    """
    sfh_type = CONFIGS[config]["sfh_type"]
    prefix = SFH_PREFIX_BY_TYPE[sfh_type]
    np.savez(
        results_dir / f"{gal_id}_{config}.npz",
        **{f"{prefix}log_total_mass": np.zeros(4)},
    )
    (results_dir / f"{gal_id}_{config}.json").write_text(json.dumps({"galaxy_id": gal_id}))


def test_an_empty_results_directory_is_refused(tmp_path):
    """No cells is not 'incomplete'; there is no figure to draw."""
    results_dir = tmp_path / "fits"
    results_dir.mkdir()
    out_dir = tmp_path / "out"

    result = _run_fig06(results_dir, out_dir)

    assert result.returncode != 0, (
        "fig06 exited 0 on an empty grid; it saved a comparison figure whose "
        f"tengri arm was empty.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert EMPTY_REFUSAL in result.stderr, (
        f"the refusal did not name the cause.\nstderr:\n{result.stderr}"
    )
    written = sorted(p.name for p in out_dir.glob("fig06*")) if out_dir.exists() else []
    assert not written, f"a figure was saved despite the refusal: {written}"


def test_a_partial_grid_is_not_refused_as_empty(tmp_path):
    """One real cell must get past the guard, stamped rather than refused.

    The run is still expected to fail here -- the stub carries none of the
    posterior arrays the figure reads, and Configuration III's stellar library
    is not present on every machine. What this test pins is *which* refusal:
    anything but the empty-grid one means the cell was seen.

    The cell has to belong to a galaxy this figure draws. Every galaxy of the
    locked sample is a panel now, so any of them would do; the first is used.
    """
    results_dir = tmp_path / "fits"
    results_dir.mkdir()
    _stub_cell(results_dir, PANEL_GALAXY_IDS[0], "III")
    out_dir = tmp_path / "out"

    result = _run_fig06(results_dir, out_dir)

    assert EMPTY_REFUSAL not in result.stderr, (
        "a grid holding one cell was refused as empty; the guard is rejecting "
        f"partial grids it should stamp.\nstderr:\n{result.stderr}"
    )


def test_a_diagnostic_array_is_not_mistaken_for_a_parameter(tmp_path):
    """Cells carry per-draw diagnostics the model has never heard of.

    fig06 selected posterior parameters by shape -- any 1-D array whose length
    matched the draw count, minus a hand-listed set of known non-parameters.
    That list has to be exhaustive to be right and was not: `divergent_mask`
    and `energy` are per-draw diagnostics of exactly that length, and they
    reached predict_properties as parameters, which refused them by name.

    Selection is now by declaration -- the model says which names are its
    parameters -- so an unrecognized array is dropped with a note rather than
    passed on. This pins that a cell carrying such arrays does not fail with
    UnknownParameterError; it may still fail for want of real posterior
    content, which is a different and honest refusal.
    """
    import sys as _sys

    _sys.path.insert(0, str(ANALYSIS_DIR))
    try:
        from config_metadata import SSP_FOR_CONFIG
    except ImportError:  # pragma: no cover - the module moved
        pytest.skip("config_metadata is not importable")

    import tengri

    try:
        tengri.load_ssp(SSP_FOR_CONFIG["III"])
    except FileNotFoundError:
        pytest.skip(
            f"{SSP_FOR_CONFIG['III']} is not on this machine; the parameter filter "
            "only runs once the model can be rebuilt"
        )

    results = tmp_path / "fits"
    results.mkdir()
    gal_id = 79
    _stub_cell(results, gal_id, "III")
    # Re-write the npz with the two diagnostics beside the sampled parameter,
    # all the same length, exactly as a real cell carries them.
    from _cell_provenance import SFH_PREFIX_BY_TYPE

    prefix = SFH_PREFIX_BY_TYPE[CONFIGS["III"]["sfh_type"]]
    n_draw = 4
    np.savez(
        results / f"{gal_id}_III.npz",
        **{
            f"{prefix}log_total_mass": np.zeros(n_draw),
            "redshift": np.full(n_draw, 1.0),
            "divergent_mask": np.zeros(n_draw),
            "energy": np.zeros(n_draw),
        },
    )
    out = tmp_path / "out"

    result = _run_fig06(results, out)

    assert "UnknownParameterError" not in result.stderr, (
        "a per-draw diagnostic reached the model as a parameter; selection is "
        f"matching on shape again rather than on the declared set.\n"
        f"stderr:\n{result.stderr[-1200:]}"
    )


def test_the_panels_are_the_locked_sample_not_a_copy_of_it():
    """The figure draws exactly the galaxies the grid fits, read from one file.

    The panels were a hand-written three-tuple from the three-galaxy version of
    the paper and outlived the move to the twenty-galaxy grid, so the one figure
    setting the grid against the published codes drew three galaxies while
    Section 7 quoted numbers over all twenty. An earlier fix taught the
    completeness guard to measure those three, which kept the guard consistent
    with the figure but left both inconsistent with the grid -- and this test's
    predecessor pinned galaxy 79 as deliberately un-drawn.

    Read from the selection file, the panels and the grid are one population,
    so the guard's question and the figure's are the same question by
    construction. Compared against the file rather than a count: a test that
    hard-coded twenty would just be the next stale copy.
    """
    assert tuple(load_expected_galaxy_ids(SELECTION_20)) == PANEL_GALAXY_IDS, (
        "fig06's panels have drifted from the locked sample; derive them from "
        "selected_galaxies_20.json rather than listing them"
    )


def test_every_published_code_in_the_catalog_has_a_marker():
    """No code the catalog reports may be silently left out of the figure.

    The marker table said "Dense Basis" while the catalog says "Dense_Basis",
    so the lookup missed on all twenty galaxies and Dense Basis was never drawn
    -- four codes under a caption naming five, with nothing to flag it. Read the
    real catalog, so a respelled or newly added code fails here rather than
    disappearing from the paper.
    """
    import fig06_code_overlay as fig06

    published = fig06.load_published_values(ANALYSIS_DIR / "results" / "art_sedfitting_z1.csv")
    reported = {code for gal in PANEL_GALAXY_IDS for code in published.get(gal, {})}
    assert reported, "the catalog reports no codes for the panel galaxies"
    missing = sorted(reported - set(fig06.CODE_MARKERS))
    assert not missing, f"codes in the catalog with no marker, so never drawn: {missing}"
