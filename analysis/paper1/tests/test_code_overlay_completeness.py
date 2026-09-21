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
from config_metadata import CONFIGS
from fig06_code_overlay import PANEL_GALAXY_IDS

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

    The cell has to belong to a galaxy this figure draws. Stubbing galaxy 79
    instead asserted that a cell for a galaxy with no panel should satisfy the
    guard, which is the defect the sibling test below now pins rather than a
    contract worth keeping.
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


def test_cells_for_galaxies_this_figure_does_not_draw_do_not_satisfy_the_guard(tmp_path):
    """A non-empty grid is not the same claim as a non-empty figure.

    The refusal above measured the locked twenty; the figure draws three of
    them. One finished cell for any of the other seventeen therefore made the
    guard pass while all three panels still had an empty tengri arm, and the
    figure saved under the paper's filename at exit 0 -- stamped with a count
    describing a grid it does not plot.

    Galaxy 79 is in the locked sample and is not a panel, so it is exactly the
    cell that used to buy a vacuous figure.
    """
    off_panel = 79
    assert off_panel not in PANEL_GALAXY_IDS, (
        "this test needs a galaxy the figure does not draw; 79 is now a panel"
    )

    results_dir = tmp_path / "fits"
    results_dir.mkdir()
    _stub_cell(results_dir, off_panel, "III")
    out_dir = tmp_path / "out"

    result = _run_fig06(results_dir, out_dir)

    assert result.returncode != 0, (
        "fig06 accepted a grid holding no cell for any galaxy it draws; "
        f"stdout:\n{result.stdout[-800:]}"
    )
    assert EMPTY_REFUSAL in result.stderr, (
        f"refused, but not for the empty tengri arm:\n{result.stderr[-1200:]}"
    )
    assert not list(out_dir.glob("fig06*")) if out_dir.exists() else True, (
        "a figure was written despite the refusal"
    )


def test_the_completeness_stamp_clears_the_legend():
    """The one line saying the grid is partial must be readable.

    This figure anchors a full-width legend below the axes. The stamp was
    placed at a fixed y just under the axes, which is inside that legend box,
    so the sentence reporting an incomplete grid was struck through by the
    legend it was drawn over -- the single element a reader most needs, and
    the only one that cannot be recovered from anywhere else in the figure.

    Measured here rather than eyeballed: a rendered check needs grid cells
    matching ``configs.py``, which are not on every machine, and a stamp that
    is verified only when someone happens to look is not verified.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fig06_code_overlay import stamp_top

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7.5))
    for ax in axes:
        ax.plot([0, 1], [0, 1], label="published code")
    legend = fig.legend(
        handles=axes[0].get_lines(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.08),
    )

    top = stamp_top(fig, legend)
    legend_bottom = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
    plt.close(fig)

    assert top < legend_bottom, (
        f"the stamp starts at y={top:.4f}, inside a legend whose bottom edge is "
        f"y={legend_bottom:.4f}; it would be drawn over the legend box"
    )
    assert top <= -0.012, (
        f"the stamp starts at y={top:.4f}, above the axes' lower edge; a legend "
        "that sits high must not push the stamp up into the figure"
    )
