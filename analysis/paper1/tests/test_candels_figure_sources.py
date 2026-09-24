# SPDX-License-Identifier: BSD-3-Clause
"""``fig05_candels_galaxies.py`` must read its sample, and must have one.

Two defects of the same shape, both of which read as green.

The panel titles print each galaxy's redshift and type label. They were
supposed to come from the committed selection; the path built to reach it
resolved to a ``results/`` directory at the repository root that has never
existed, and a fallback beneath it substituted a literal written into the
figure script. The fallback therefore won every run this figure has ever had,
and the selection could not reach the figure it names. The redshifts agreed,
so nothing wrong was published -- but agreement discovered after the fact is
not a mechanism.

Second, every panel of this figure is tengri output, and it had no refusal for
a results directory holding none. fig06 shipped precisely that figure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parents[1]
FIG05 = ANALYSIS_DIR / "fig05_candels_galaxies.py"
SELECTION = ANALYSIS_DIR / "results" / "selected_galaxies.json"

sys.path.insert(0, str(ANALYSIS_DIR))

pytestmark = pytest.mark.contract


def test_panel_metadata_is_read_from_the_committed_selection():
    """Not from a literal in the figure script.

    Compared field by field against the file rather than against remembered
    values: a test that restated the expected redshifts here would pass on the
    fallback too, which is how this went unnoticed.
    """
    from fig05_candels_galaxies import GALAXY_IDS, load_galaxy_metadata

    declared = {
        entry["id"]: entry for entry in json.loads(SELECTION.read_text())["selected_galaxies"]
    }
    loaded = load_galaxy_metadata()

    for gal_id in GALAXY_IDS:
        assert gal_id in loaded, f"{gal_id} is drawn but carries no metadata"
        assert loaded[gal_id]["z"] == declared[gal_id]["z"], (
            f"galaxy {gal_id} is drawn at z={loaded[gal_id]['z']} while the "
            f"selection declares z={declared[gal_id]['z']}"
        )
        assert loaded[gal_id]["class"] == declared[gal_id]["type_label"].replace("_", " "), (
            f"galaxy {gal_id} is labeled {loaded[gal_id]['class']!r} while the "
            f"selection declares {declared[gal_id]['type_label']!r}; the figure "
            "is reading something other than the selection"
        )


def test_an_empty_results_directory_is_refused(tmp_path):
    """Axes with nothing in them is not a figure."""
    results_dir = tmp_path / "fits"
    results_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(FIG05),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--results-output-dir",
            str(tmp_path / "sidecar"),
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

    assert result.returncode != 0, (
        f"fig05 drew a figure from an empty grid.\nstdout:\n{result.stdout[-800:]}"
    )
    assert "no finished cells" in result.stderr, (
        f"refused, but not for the empty grid:\n{result.stderr[-1200:]}"
    )
    assert not list((tmp_path / "out").glob("fig05*")) if (tmp_path / "out").exists() else True
