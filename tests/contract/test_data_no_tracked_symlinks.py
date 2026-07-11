# SPDX-License-Identifier: BSD-3-Clause
"""No tracked path under ``data/`` may be a symlink.

Guards the accident class that broke Cue-out-of-the-box twice: a worktree
convenience symlink (``ln -sf ~/canonical/data/... data/``) placed over a
*tracked* data file gets staged by a broad ``git add``, and the squash-merge
ships a 58-byte absolute-path symlink to every clone in place of the real
data (#955 carried ``cue_weights.npz`` + ``fsps_mass_remaining_chabrier.h5``;
an older commit carried ``dl07_raw``/``dl14_raw``/``synthesizer_grids``).
A dangling machine-specific link is worse than a missing file: loaders see an
existing path and fail with confusing errors instead of the loud
FileNotFoundError contract.
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_REPO = Path(__file__).resolve().parents[2]


def test_no_tracked_symlinks_under_data():
    out = subprocess.run(
        ["git", "ls-files", "-s", "--", "data/"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout (sdist/wheel install)")
    offenders = [
        line.split("\t", 1)[1] for line in out.stdout.splitlines() if line.split()[0] == "120000"
    ]
    assert not offenders, (
        f"Tracked symlinks under data/ (machine-specific, dangling on every "
        f"other clone): {offenders}. Commit the real file or untrack it."
    )
