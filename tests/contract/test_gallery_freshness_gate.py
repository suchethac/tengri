# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: the gallery freshness gate and the scoped-regen fence.

Two things are pinned here, and they only work as a pair (#805).

**The gate** (``tools/check_gallery_fresh.py``) reports examples whose
committed ``docs/auto_examples/`` render has drifted from the source. It ran
warn-only for months, so drift accumulated silently: 16 examples in July,
60 by August. A gate that cannot fail is not a gate.

**The fence** (``tools/regen_gallery.py``) is what makes the gate satisfiable.
A plain ``make html`` — the remedy ``docs/conf.py`` itself recommends —
*degrades* the gallery (#1236): every example with a committed figure is
skipped, yet every ``.rst`` is rewritten, so pages come back stripped of
everything execution produced. Measured on a single-example build: 195 files
changed, 45,204 deletions. The fence restores every page that was not a
regeneration target, which is the only reason a scoped regen is safe.

Pointing a hard gate at a remedy that silently destroys the artifact would be
worse than no gate, so the fence's correctness is a contract, not a detail.
"""

import hashlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "tests.yml"


def _load(name: str):
    """Import a ``tools/`` script by path — they are not an installed package."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fresh = _load("check_gallery_fresh")
regen = _load("regen_gallery")


# --------------------------------------------------------------------------
# The gate runs, and runs strictly
# --------------------------------------------------------------------------


def test_ci_runs_the_freshness_gate_strictly():
    """The workflow must invoke the gate with ``--strict``.

    Read out of the workflow rather than restated here, so dropping the flag
    fails this test instead of quietly restoring warn-only mode.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    calls = re.findall(r"^\s*-?\s*run:\s*(python\s+tools/check_gallery_fresh\.py.*)$", text, re.M)
    assert calls, "the freshness gate is not run by .github/workflows/tests.yml at all"
    for call in calls:
        assert "--strict" in call, f"gate runs warn-only, so drift cannot fail CI: {call!r}"


def test_committed_gallery_is_fresh():
    """No committed render may lag its source.

    This is the invariant ``--strict`` enforces in CI; asserting it here too
    means the fast tier says *which* examples drifted rather than only that a
    lint job went red.
    """
    stale, unrendered = fresh.stale_examples()
    assert not stale, f"{len(stale)} example(s) drifted from their committed render: {stale}"
    assert not unrendered, f"{len(unrendered)} example(s) have no committed render: {unrendered}"


# --------------------------------------------------------------------------
# The detector is not vacuous
# --------------------------------------------------------------------------


def _synthetic_gallery(tmp_path: Path) -> tuple[Path, Path]:
    """A two-example tree: one render current, one drifted, one unrendered."""
    examples, auto = tmp_path / "examples", tmp_path / "auto_examples"
    (examples / "sec").mkdir(parents=True)
    (auto / "sec").mkdir(parents=True)

    current = examples / "sec" / "plot_current.py"
    current.write_text("x = 1\n")
    stamp = hashlib.md5(current.read_bytes()).hexdigest()
    (auto / "sec" / "plot_current.py.md5").write_text(stamp)

    drifted = examples / "sec" / "plot_drifted.py"
    drifted.write_text("y = 2\n")
    (auto / "sec" / "plot_drifted.py.md5").write_text("0" * 32)

    (examples / "sec" / "plot_unrendered.py").write_text("z = 3\n")
    return examples, auto


def test_detector_separates_drifted_from_current(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(fresh, "AUTO", tmp_path / "auto_examples")
    _synthetic_gallery(tmp_path)

    stale, unrendered = fresh.stale_examples()

    assert stale == ["sec/plot_drifted.py"]
    assert unrendered == ["sec/plot_unrendered.py"]


def test_strict_fails_on_drift_and_warn_only_does_not(tmp_path, monkeypatch):
    """The flag is the whole difference between a report and a gate."""
    monkeypatch.setattr(fresh, "EXAMPLES", tmp_path / "examples")
    monkeypatch.setattr(fresh, "AUTO", tmp_path / "auto_examples")
    _synthetic_gallery(tmp_path)

    monkeypatch.setattr(sys, "argv", ["check_gallery_fresh.py", "--strict"])
    assert fresh.main() == 1

    monkeypatch.setattr(sys, "argv", ["check_gallery_fresh.py"])
    assert fresh.main() == 0


# --------------------------------------------------------------------------
# The fence: ownership is exact, never by prefix
# --------------------------------------------------------------------------


def test_fence_ownership_is_exact_not_prefix():
    """A longer example's files are not owned by a shorter target.

    Synthetic names, so this cannot go vacuous if the repo is renamed. The
    real pair is covered below.
    """
    targets = {"plot_alpha"}
    owned = "x/plot_alpha.rst", "x/plot_alpha.py.md5", "x/images/sphx_glr_plot_alpha_001.png"
    foreign = (
        "x/plot_alpha_beta.rst",
        "x/plot_alpha_beta.py.md5",
        "x/images/sphx_glr_plot_alpha_beta_001.png",
    )
    for path in owned:
        assert regen._owned_by_targets(Path(path), targets), path
    for path in foreign:
        assert not regen._owned_by_targets(Path(path), targets), path


def _nesting_pairs() -> list[tuple[str, str]]:
    names = sorted(p.stem for p in (_ROOT / "examples").rglob("plot_*.py"))
    return [(a, b) for a in names for b in names if a != b and b.startswith(a + "_")]


@pytest.mark.parametrize(("shorter", "longer"), _nesting_pairs())
def test_fence_does_not_claim_a_nesting_siblings_files(shorter, longer):
    """Every real basename that nests inside another, checked against the fence."""
    assert regen._owned_by_targets(Path(f"a/{shorter}.rst"), {shorter})
    assert not regen._owned_by_targets(Path(f"a/{longer}.rst"), {shorter})


# --------------------------------------------------------------------------
# The fence: what a scoped build keeps, restores, and removes
# --------------------------------------------------------------------------


def test_fence_keeps_targets_restores_others_and_removes_strays(tmp_path, monkeypatch):
    """The three outcomes a scoped regen depends on, in one pass.

    Models exactly what a build does: the target page is rewritten *better*
    (it executed), an untargeted page is rewritten *hollow* (it was skipped),
    and a stray file appears. Only the first may survive.
    """
    auto = tmp_path / "auto_examples"
    (auto / "sec").mkdir(parents=True)
    target = auto / "sec" / "plot_target.rst"
    other = auto / "sec" / "plot_other.rst"
    target.write_text("old target render")
    other.write_text("good render with executed output")

    snapshot = tmp_path / "snapshot"
    shutil.copytree(auto, snapshot)

    target.write_text("new target render")
    other.write_text("")  # hollowed out by the skipped-but-rewritten build
    (auto / "sec" / "plot_stray.rst").write_text("page for an example that no longer exists")

    monkeypatch.setattr(regen, "AUTO", auto)
    restored, removed, kept = regen._restore_untargeted(snapshot, {"plot_target"})

    assert target.read_text() == "new target render", "the regenerated target was clobbered"
    assert other.read_text() == "good render with executed output", "collateral damage survived"
    assert not (auto / "sec" / "plot_stray.rst").exists(), "stray page left behind"
    assert (restored, removed, kept) == (1, 1, 1)


def test_fence_restores_whole_build_timing_table(tmp_path, monkeypatch):
    """``sg_execution_times.rst`` describes a whole build, so a scoped one lies.

    It is captioned "for 279 files" but rewritten from only what this build
    executed, and it lives outside ``auto_examples/`` where the page fence does
    not reach. Left alone, a six-example batch publishes its own runtime as the
    gallery's total.
    """
    auto = tmp_path / "auto_examples"
    auto.mkdir(parents=True)
    times = tmp_path / "sg_execution_times.rst"
    times.write_text("**00:01.561** total execution time for 279 files")

    snapshot = tmp_path / "snap" / "auto_examples"
    snapshot.parent.mkdir()
    monkeypatch.setattr(regen, "AUTO", auto)
    monkeypatch.setattr(regen, "BUILD_ARTIFACTS", (times,))
    regen._snapshot(snapshot)

    times.write_text("**02:00.574** total execution time for 279 files")
    assert regen._restore_build_artifacts(snapshot) == 1
    assert "00:01.561" in times.read_text()
