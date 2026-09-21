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
    """
    results_dir = tmp_path / "fits"
    results_dir.mkdir()
    _stub_cell(results_dir, 79, "III")
    out_dir = tmp_path / "out"

    result = _run_fig06(results_dir, out_dir)

    assert EMPTY_REFUSAL not in result.stderr, (
        "a grid holding one cell was refused as empty; the guard is rejecting "
        f"partial grids it should stamp.\nstderr:\n{result.stderr}"
    )
