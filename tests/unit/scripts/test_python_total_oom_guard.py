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


def _write_pressure(path: Path, avail_kb: int, swap_used_kb) -> Path:
    """Write one pressure sample per tick.

    ``swap_used_kb`` may be a sequence to drive a ramp across ticks; a scalar is
    a constant. The guard holds the last line once the fixture is exhausted.
    """
    samples = swap_used_kb if isinstance(swap_used_kb, (list, tuple)) else [swap_used_kb]
    path.write_text("".join(f"{avail_kb} {s}\n" for s in samples))
    return path


def _run_guard(
    tmp_path: Path, ps_procs, *, avail_kb, swap_used_kb, shipped_all_params=False, **env_overrides
):
    """Run one guard tick against fixtures and return the log text.

    ``shipped_all_params=True`` omits the threshold neutralizers so the guard runs
    the configuration a real machine gets. Any test asking "would the guard as
    installed have caught this?" MUST use it: setting the thresholds by hand
    proves only that some configuration works, never that the shipped default
    does — and a wrong default was the whole of the 2026-08-10 miss.
    """
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
    }
    if not shipped_defaults:
        # Neutralize the pressure triggers unless a test opts in. SWAP_MAX_GB=0
        # also disables the avail-soft conjunction, which requires both halves.
        env.update({"AVAIL_PCT_MIN": "0", "SWAP_MAX_GB": "0", "SWAP_GROWTH_GB": "0"})
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

    The fixture must contain the "something else", or the claim is untestable:
    with only python in the process table python is 100% of resident memory and
    suppressing is the wrong answer. The 40 GB hog below is the elsewhere.
    """
    procs = [
        (701, 1 * GB, "/venv/bin/python tiny-a"),
        (702, 1 * GB, "/venv/bin/python tiny-b"),
        (800, 40 * GB, "/Applications/Hog.app/Contents/MacOS/Hog"),
    ]
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
    assert "Hog" in log, f"the log must name the culprit, not just say 'elsewhere':\n{log}"


def test_swap_undercount_must_not_veto_a_firing_trigger(tmp_path):
    """The 2026-08-16 crash: summed RSS vetoed six consecutive growth trips.

    Two python processes held 20 GB+ each, but most of their pages were swapped
    out, so RSS summed to 4.98 GB — under ``PRESSURE_MIN_PYTHON_GB=8``. The gate
    concluded "the memory is elsewhere", suppressed every trip, and the box
    panicked ~4 minutes later on a watchdogd timeout.

    RSS is the number that collapses *because* the machine is thrashing, which
    is why it was demoted as a trigger — and it must not be allowed to veto the
    triggers that did fire. Python's *share* of resident memory survives the
    undercount, because everything resident shrinks together.
    """
    procs = [
        (901, 2500 * MB, "/venv/bin/python -m pytest heavy-a"),
        (902, 2000 * MB, "/venv/bin/python -m pytest heavy-b"),
        (903, 480 * MB, "/venv/bin/python -u -c import sys"),
    ]
    log = _run_guard(
        tmp_path,
        procs,
        avail_kb=int(0.18 * 48 * GB),
        swap_used_kb=[15 * GB, 18 * GB, 21 * GB, 22 * GB],
        shipped_all_params=True,
        RAM_KB_OVERRIDE=48 * GB,
        TOTAL_LIMIT_GB=32,
        MAX_TICKS=4,
    )

    assert "TRIP" in log, f"the growth trigger was vetoed by an undercounted RSS again:\n{log}"
    assert _victim_pids(log), "tripped but shed nobody"


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


# The machine measured during the 2026-08-10 incident: 48 GiB of RAM, 34 python
# processes whose *resident* pages summed to only 12.88 GB because the rest had
# been paged out, available memory pinned at 18%, swap at 43 GB. Every trigger
# as shipped was structurally unable to fire, and the user killed by hand.
_RAM_KB_48 = 48 * GB
_INCIDENT_AVAIL_KB = int(0.18 * _RAM_KB_48)
_INCIDENT_SWAP_KB = 43 * GB
_INCIDENT_PROCS = [(600 + i, 388 * MB, f"/venv/bin/python -m pytest w{i}") for i in range(34)]


def test_the_2026_08_10_incident_now_trips(tmp_path):
    """Available 18% + swap 43 GB must shed, though no single threshold is hit.

    Neither half fires alone by design: 18% clears the 10% hard floor, and the
    conjunction is what makes the pair sensitive without making either jumpy.
    """
    log = _run_guard(
        tmp_path,
        _INCIDENT_PROCS,
        avail_kb=_INCIDENT_AVAIL_KB,
        swap_used_kb=_INCIDENT_SWAP_KB,
        shipped_all_params=True,
        RAM_KB_OVERRIDE=_RAM_KB_48,
        TOTAL_LIMIT_GB=32,
    )

    assert "TRIP" in log, f"guard stayed silent through the incident:\n{log}"
    assert _victim_pids(log), "tripped but selected nobody"


def test_the_incident_does_not_trip_the_sum_rss_limit(tmp_path):
    """The sum-RSS trigger alone is blind here — that is why it cannot be the only one.

    12.88 GB of resident pages against a 32 GB limit. Pinning this keeps the
    regression honest: if someone 'fixes' the miss by lowering TOTAL_LIMIT_GB,
    this test still shows the RSS axis never saw the emergency.
    """
    log = _run_guard(
        tmp_path,
        _INCIDENT_PROCS,
        avail_kb=_INCIDENT_AVAIL_KB,
        swap_used_kb=_INCIDENT_SWAP_KB,
        RAM_KB_OVERRIDE=_RAM_KB_48,
        TOTAL_LIMIT_GB=32,
    )

    assert "TRIP" not in log
    assert "sum-rss" not in log


def test_a_recovered_machine_holding_more_rss_does_not_trip(tmp_path):
    """The control, and the point: MORE resident python, but healthy — stay quiet.

    Ten processes summing to 23.8 GB with 44% available and swap back at 8.4 GB
    is the *post-rescue* reading from the same box. A guard driven by resident
    memory would shed here and not during the incident, i.e. exactly backwards.
    """
    procs = [(700 + i, int(2.38 * GB), f"/venv/bin/python -m pytest w{i}") for i in range(10)]
    log = _run_guard(
        tmp_path,
        procs,
        avail_kb=int(0.44 * _RAM_KB_48),
        swap_used_kb=int(8.4 * GB),
        shipped_all_params=True,
        RAM_KB_OVERRIDE=_RAM_KB_48,
        TOTAL_LIMIT_GB=32,
    )

    assert "TRIP" not in log, f"shed on a healthy machine:\n{log}"


def test_swap_growth_trips_below_the_absolute_level(tmp_path):
    """A fast climb is an emergency at any baseline — level triggers arrive late.

    Swap ramps 6 -> 17 GB while staying under SWAP_MAX_GB=20 the whole time,
    which is the real 00:53->00:57 ramp. Available stays at 30%, above both the
    hard floor and the soft threshold, so growth is the only thing that can fire.
    """
    log = _run_guard(
        tmp_path,
        _INCIDENT_PROCS,
        avail_kb=int(0.30 * _RAM_KB_48),
        swap_used_kb=[6 * GB, 9 * GB, 13 * GB, 17 * GB],
        shipped_all_params=True,
        RAM_KB_OVERRIDE=_RAM_KB_48,
        TOTAL_LIMIT_GB=32,
        MAX_TICKS=4,
    )

    assert "TRIP" in log, f"missed an 11 GB swap ramp:\n{log}"
    assert "swap +" in log


def test_flat_high_swap_alone_does_not_trip(tmp_path):
    """The trap the disabled trigger was guarding against, kept closed.

    Swap parked at 25 GB with plenty of memory free is a box that swapped once
    and recovered. Neither the conjunction (available is high) nor growth (flat)
    may fire, or the guard kills 8 GB out of every long-running job.
    """
    log = _run_guard(
        tmp_path,
        _INCIDENT_PROCS,
        avail_kb=int(0.40 * _RAM_KB_48),
        swap_used_kb=[25 * GB, 25 * GB, 25 * GB],
        shipped_all_params=True,
        RAM_KB_OVERRIDE=_RAM_KB_48,
        TOTAL_LIMIT_GB=32,
        MAX_TICKS=3,
    )

    assert "TRIP" not in log, f"fired on flat, recovered swap:\n{log}"
