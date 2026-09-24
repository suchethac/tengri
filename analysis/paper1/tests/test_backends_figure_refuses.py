# SPDX-License-Identifier: BSD-3-Clause
"""``fig07_backends`` must refuse a sweep with no per-draw arrays, not crash.

Each backend in the sweep writes two files: a ``.json`` summary and a ``.npz``
of per-draw values. Only the summaries were tracked, so on any checkout
without the untracked originals the figure logged "Skipping <backend>: NPZ not
found" five times -- once per guarded read -- and then died on an unguarded
sixth read of the same absence, at ``np.load(sweep_dir / "map.npz")``.

Two different things were wrong and only one of them was the crash. The same
missing file was tolerated in one code path and fatal in the next, which is
the shape worth testing for: a figure that half-succeeds and then raises has
already told you it can cope, so the traceback reads as a bug in the data
rather than in the script.

The arrays are committed now, so this situation should not arise from a clean
checkout. The test pins the behavior anyway, because "it cannot happen" is
what was true of the tracked ``.json`` files too.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parents[1]
FIG07 = ANALYSIS_DIR / "fig07_backends.py"
SWEEP = ANALYSIS_DIR / "results" / "backend_sweep_pin"

pytestmark = pytest.mark.contract

REFUSAL = "no per-draw NPZ"


def _run(sweep_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FIG07), "--sweep-dir", str(sweep_dir), "--out-dir", str(out_dir)],
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


def test_the_per_draw_arrays_are_committed():
    """The fix for the reader; the tests below are the fix for the script."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(SWEEP.relative_to(REPO_ROOT))],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.split()
    npz = sorted(Path(t).name for t in tracked if t.endswith(".npz"))
    assert npz, (
        "no .npz is tracked under backend_sweep_pin, so a reader with only this "
        "repository cannot rebuild fig07 -- which is the whole claim the "
        "paper1_figures README makes"
    )


def test_a_sweep_with_no_arrays_is_refused_rather_than_crashing(tmp_path):
    """Summaries present, arrays absent: the state a bare checkout used to be."""
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    for name in ("map", "laplace", "mcmc_hmc", "mcmc_nuts_fast", "nss"):
        source = SWEEP / f"{name}.json"
        if not source.is_file():
            pytest.skip(f"committed summary missing: {source.name}")
        (sweep / f"{name}.json").write_text(source.read_text())
    out = tmp_path / "out"

    result = _run(sweep, out)

    assert result.returncode != 0, "a sweep with no per-draw arrays was accepted"
    assert REFUSAL in result.stderr, (
        f"the refusal did not name the cause.\nstderr:\n{result.stderr[-1500:]}"
    )
    # The discriminator between the fix and the bug: a refusal explains, a
    # crash produces a traceback after having already skipped the same files.
    assert "Traceback" not in result.stderr, (
        "fig07 crashed rather than refusing; the unguarded map.npz read is "
        f"back.\nstderr:\n{result.stderr[-1500:]}"
    )
    written = sorted(p.name for p in out.glob("fig07*")) if out.exists() else []
    assert not written, f"a figure was written despite the refusal: {written}"


def test_a_sweep_missing_only_map_still_draws(tmp_path):
    """MAP is one point on a panel of marginals, not a precondition.

    Guards the fix from being over-applied: build_figure already treats "map"
    as optional, so refusing the whole figure for it alone would throw away
    four backends that are present.
    """
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    present = 0
    for name in ("map", "laplace", "mcmc_hmc", "mcmc_nuts_fast", "nss"):
        if (SWEEP / f"{name}.json").is_file():
            (sweep / f"{name}.json").write_text((SWEEP / f"{name}.json").read_text())
        npz = SWEEP / f"{name}.npz"
        if name != "map" and npz.is_file():
            data = np.load(npz, allow_pickle=False)
            np.savez(sweep / f"{name}.npz", **{k: data[k] for k in data.files})
            present += 1
    if present == 0:
        pytest.skip("no per-draw arrays available to build the partial sweep from")
    out = tmp_path / "out"

    result = _run(sweep, out)

    assert REFUSAL not in result.stderr, (
        "a sweep holding four backends' draws was refused for want of MAP's "
        f"single point.\nstderr:\n{result.stderr[-1200:]}"
    )
