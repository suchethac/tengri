# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/check_build_listings.py.

Covers the executed-listing gate that stops a documented ``SEDModel.build``
call from naming a nonexistent kwarg (#2430). Verifies that:
1. A valid listing executes and its sampled photometry is finite and positive
2. A broken listing is reported with its file and fence line, and the exit code is 1
3. ``--list`` names every listing it finds and exits 0
4. A listing marked ``<!-- raises: Name -->`` passes only by raising ``Name``
5. ``--one PATH:LINE`` runs a single listing and prints its result line
6. A listing that runs past ``--timeout`` is reported as a timeout failure
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "check_build_listings.py"

VALID_LISTING = """\
Build a model from the fixture grid and sample it.

```python
model = SEDModel.build(ssp, observation=obs, redshift=0.1)
```
"""

RAISES_LISTING = """\
A deliberate refusal, documented as such.

<!-- raises: ParameterError -->
```python
model = SEDModel.build(
    ssp, observation=obs,
    radio={'sf': {'type': 'none'}, 'agn': {'type': 'none'}, 'all_params': FREE},
    redshift=Fixed(0.1),
)
```
"""

SLOW_LISTING = """\
A listing that does not finish within the timeout.

```python
import time
time.sleep(3)
model = SEDModel.build(ssp, observation=obs, redshift=0.1)
```
"""

BROKEN_LISTING = """\
A listing with a kwarg that does not exist.

```python
model = SEDModel.build(ssp, observation=obs, redshfit=0.1, no_such_kwarg=True)
```
"""


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "listings.md"
    path.write_text(VALID_LISTING + "\n" + BROKEN_LISTING, encoding="utf-8")
    return path


def test_broken_listing_is_reported_with_its_fence_line(doc: Path):
    result = _run("--no-default", "--paths", str(doc))

    # Valid fence opens at line 3; the broken one opens at line 9.
    broken_line = 9
    assert result.returncode == 1, result.stdout + result.stderr
    failure_lines = [ln for ln in result.stdout.splitlines() if ln.startswith(str(doc))]
    assert len(failure_lines) == 1, result.stdout
    assert failure_lines[0].startswith(f"{doc}:{broken_line}: ")
    assert "2 listings, 2 executed, 1 failed" in result.stdout


def test_one_runs_a_single_listing_and_prints_ok(doc: Path):
    result = _run("--one", f"{doc}:3")

    assert result.returncode == 0, result.stdout + result.stderr
    assert any(ln.startswith("ok ") for ln in result.stdout.splitlines()), result.stdout


def test_listing_past_timeout_is_reported_as_timeout(tmp_path: Path):
    path = tmp_path / "slow.md"
    path.write_text(SLOW_LISTING, encoding="utf-8")

    result = _run("--no-default", "--paths", str(path), "--timeout", "1")

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:3: timeout after 1 s" in result.stdout, result.stdout


def test_list_names_every_listing_and_exits_zero(doc: Path):
    result = _run("--list", "--no-default", "--paths", str(doc))

    assert result.returncode == 0, result.stderr
    assert f"{doc}:3" in result.stdout
    assert f"{doc}:9" in result.stdout


def test_raises_marker_passes_only_on_the_named_error(tmp_path: Path):
    path = tmp_path / "refusal.md"
    path.write_text(RAISES_LISTING, encoding="utf-8")

    result = _run("--no-default", "--paths", str(path))

    # The refusal raises ParameterError as documented, so the page passes.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 listings, 1 executed, 0 failed" in result.stdout
