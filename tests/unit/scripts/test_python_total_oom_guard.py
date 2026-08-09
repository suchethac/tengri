# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for ``scripts/python_total_oom_guard.sh``.

The guard is a shell daemon, so these drive it through its two test hooks:

``PS_FIXTURE``
    a file of ``pid rss_kb args`` lines standing in for ``ps -axo pid=,rss=,args=``
``PRESSURE_FIXTURE``
    a file of ``avail_kb swap_used_kb`` standing in for ``vm_stat``/``sysctl``

Each test runs one tick with ``MAX_TICKS=1 DRY_RUN=1`` and reads the kill plan
out of the log, so nothing on the developer's machine is ever signaled.

Incident these pin (2026-08-09): a 48 GB machine reached ~120 GB of summed
python RSS and the guard shed nothing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[3] / "scripts" / "python_total_oom_guard.sh"

MB = 1024  # KB in a MB
GB = 1024 * 1024  # KB in a GB


def _write_ps(path: Path, procs: list[tuple[int, int, str]]) -> Path:
    """Write a fake ``ps`` table. ``procs`` is ``(pid, rss_kb, args)``."""
    path.write_text("".join(f"{pid:>8} {rss:>10} {args}\n" for pid, rss, args in procs))
    return path


def _write_pressure(path: Path, avail_kb: int, swap_used_kb: int) -> Path:
    path.write_text(f"{avail_kb} {swap_used_kb}\n")
    return path


def _run_guard(tmp_path: Path, ps_procs, *, avail_kb, swap_used_kb, **env_overrides):
    """Run one guard tick against fixtures and return the log text."""
    log = tmp_path / "guard.log"
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PS_FIXTURE": str(_write_ps(tmp_path / "ps.txt", ps_procs)),
        "PRESSURE_FIXTURE": str(
            _write_pressure(tmp_path / "pressure.txt", avail_kb, swap_used_kb)
        ),
        "LOG": str(log),
        "DRY_RUN": "1",
        "MAX_TICKS": "1",
        "INTERVAL_SEC": "1",
        # Neutralize the pressure triggers unless a test opts in.
        "AVAIL_PCT_MIN": "0",
        "SWAP_MAX_GB": "0",
    }
    env.update({k: str(v) for k, v in env_overrides.items()})
    proc = subprocess.run(
        ["bash", str(GUARD)], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"guard exited {proc.returncode}\n{proc.stderr}"
    return log.read_text()


def _victim_pids(log: str) -> list[int]:
    """Pids in the order the guard chose them."""
    return [
        int(line.split("pid=")[1].split()[0])
        for line in log.splitlines()
        if "would SIGKILL" in line
    ]


@pytest.fixture(autouse=True)
def _require_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    if not GUARD.exists():
        pytest.skip(f"guard script not found at {GUARD}")


def test_many_small_processes_are_shed_despite_min_kill_floor(tmp_path):
    """A MIN_KILL_MB floor must be a preference, not a veto.

    This is the defect that made the guard a no-op on a real workload: 200
    workers of 300 MB sum to ~58 GB, but every one of them is below the
    512 MB floor, so a single-pass selector picks nobody and the machine dies
    with the guard running and 'healthy'.
    """
    procs = [(2000 + i, 300 * MB, f"/venv/bin/python worker-{i}") for i in range(200)]
    log = _run_guard(
        tmp_path, procs, avail_kb=40 * GB, swap_used_kb=0, TOTAL_LIMIT_GB=30, MIN_KILL_MB=512
    )

    victims = _victim_pids(log)
    assert victims, "guard selected no victims — the floor vetoed every candidate"
    # 58.6 GB present, 30 GB limit => must shed ~28.6 GB => ~98 processes.
    assert len(victims) >= 90, f"shed only {len(victims)} of ~98 needed"


def test_victims_are_selected_largest_first(tmp_path):
    """Shedding must start with the most memory-hungry process."""
    procs = [
        (101, 1 * GB, "/venv/bin/python small"),
        (102, 6 * GB, "/venv/bin/python biggest"),
        (103, 3 * GB, "/venv/bin/python middle"),
    ]
    # 10 GB present, 5 GB limit => shed 5 GB => biggest (6 GB) alone suffices.
    log = _run_guard(tmp_path, procs, avail_kb=40 * GB, swap_used_kb=0, TOTAL_LIMIT_GB=5)

    assert _victim_pids(log) == [102]


def test_pressure_trips_even_when_sum_rss_is_under_the_limit(tmp_path):
    """The sum-RSS limit can sit above what the workload ever reaches.

    Observed on the incident machine: python summed to 12-16 GB against a
    30 GB limit — never tripping — while swap climbed past 23 GB and the box
    thrashed. Real OS pressure must be sufficient on its own.
    """
    procs = [(300 + i, 2 * GB, f"/venv/bin/python worker-{i}") for i in range(6)]
    log = _run_guard(
        tmp_path,
        procs,
        avail_kb=1 * GB,  # 1 GB available of 48 GB
        swap_used_kb=23 * GB,
        TOTAL_LIMIT_GB=30,  # 12 GB total: sum-RSS trigger stays silent
        SWAP_MAX_GB=12,
        SHED_GB=8,
    )

    assert "sum-rss" not in log, "sum-rss should not have tripped at 12 GB under a 30 GB limit"
    assert "swap" in log and "TRIP" in log
    victims = _victim_pids(log)
    assert len(victims) == 4, f"expected 4 x 2 GB to meet the 8 GB shed target, got {victims}"


def test_pressure_gate_suppresses_shedding_when_python_is_small(tmp_path):
    """Killing python must not become superstition.

    Swap pressure is often chronic and is not always python's fault. Without a
    gate the guard re-trips every cooldown and eventually kills every python
    process, because 2 GB of python cannot fix a 20 GB shortfall caused by
    something else. The gate also makes the runaway self-limiting: once enough
    has been shed, python falls under it and shedding stops on its own.
    """
    procs = [(701, 1 * GB, "/venv/bin/python tiny-a"), (702, 1 * GB, "/venv/bin/python tiny-b")]
    log = _run_guard(
        tmp_path,
        procs,
        avail_kb=1 * GB,
        swap_used_kb=23 * GB,
        TOTAL_LIMIT_GB=30,
        SWAP_MAX_GB=12,
        AVAIL_PCT_MIN=15,
        PRESSURE_MIN_PYTHON_GB=8,
    )

    assert _victim_pids(log) == []
    assert "not shedding" in log


def test_healthy_machine_kills_nothing(tmp_path):
    """No trigger, no kills — the guard must not shed on a quiet machine."""
    procs = [(400 + i, 1 * GB, f"/venv/bin/python worker-{i}") for i in range(4)]
    log = _run_guard(
        tmp_path,
        procs,
        avail_kb=30 * GB,
        swap_used_kb=0,
        TOTAL_LIMIT_GB=30,
        AVAIL_PCT_MIN=15,
        SWAP_MAX_GB=12,
    )

    assert _victim_pids(log) == []
    assert "TRIP" not in log


def test_excluded_processes_are_never_shed(tmp_path):
    """EXCLUDE_RE must survive the floor-ignoring second pass."""
    procs = [
        (501, 9 * GB, "/System/Library/python protected"),
        (502, 4 * GB, "/venv/bin/python sheddable"),
    ]
    log = _run_guard(
        tmp_path,
        procs,
        avail_kb=40 * GB,
        swap_used_kb=0,
        TOTAL_LIMIT_GB=1,
        EXCLUDE_RE="^/System/",
    )

    assert 501 not in _victim_pids(log)
    assert 502 in _victim_pids(log)
