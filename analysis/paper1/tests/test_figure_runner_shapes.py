# SPDX-License-Identifier: BSD-3-Clause
"""``paper1_figures/_run.py`` against the script shape that has no ``main``.

The figure scripts in this directory do not share a CLI contract. Three of the
four shapes hang their arguments off a ``main``; the fourth, currently
``fig03_precompute``, parses ``sys.argv`` and writes its figures at **module
scope**, so the work happens during the import itself.

That shape breaks two assumptions a runner normally gets away with, and both
failures are silent:

1. patching ``sys.argv`` around the ``main`` call is too late -- by then the
   module has already read the notebook's own arguments and written its
   figures wherever its defaults point;
2. Python imports a module once per process, so a second call does nothing at
   all while reporting success.

Both are exercised here against the real script and its committed data rather
than a stand-in, because a stand-in would be written to the runner's
assumptions instead of the script's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "paper1_figures"))

pytestmark = pytest.mark.contract

MODULE_SCOPE_SCRIPT = "fig03_precompute"
#: The figure the appendix prints, and so the one worth asserting on.
EXPECTED = "figB1_lut_accuracy.pdf"
BENCH = ANALYSIS_DIR / "results" / "fig03_bench_forward_2026-08-30.json"
ACCURACY = ANALYSIS_DIR / "results" / "fig03_precompute_data.json"


@pytest.fixture(scope="module")
def run_figure():
    from _run import run_figure as fn

    return fn


def _argv(out_dir: Path) -> list[str]:
    return [
        "--bench-json",
        str(BENCH),
        "--accuracy-json",
        str(ACCURACY),
        "--out-dir",
        str(out_dir),
    ]


@pytest.fixture(autouse=True)
def _require_inputs():
    for path in (BENCH, ACCURACY):
        if not path.is_file():
            pytest.skip(f"committed input missing: {path.name}")


@pytest.fixture(autouse=True)
def _fresh_import():
    """Let each test decide its own import history.

    Without this, whichever test imported the script first left it in
    ``sys.modules`` and every later call took the runner's reload path. That
    masks an argv patched after the import completely, because the reload
    happens after the patch under either ordering -- the bug only bites on the
    genuinely first import in a process. Caught by mutation: restoring the old
    ordering left these tests green until this fixture existed.
    """
    name = f"analysis.paper1.{MODULE_SCOPE_SCRIPT}"
    sys.modules.pop(name, None)
    yield
    sys.modules.pop(name, None)


def test_the_script_really_has_no_main(run_figure, monkeypatch):
    """Guards the two below: if it grows a main they test a different path.

    Not a style assertion -- the runner branches on this, so a `main` appearing
    silently moves these tests onto a code path they were not written for and
    they would keep passing without covering the module-scope shape at all.

    **The import needs a patched argv**, and leaving it out is why this test
    spent its life passing for the wrong reason. The script parses at module
    scope -- that is the shape being guarded -- so importing it runs argparse
    against whatever ``sys.argv`` holds, which under pytest is pytest's own
    command line and exits 2 on the first unrecognized argument. The suite
    bakes ``-n auto`` into addopts, and xdist workers carry a clean argv, so
    the hostile case never arose: this passes in isolation, across its own
    file, and across all 230 tests of the directory, and fails immediately
    under ``-n 0``. Do not remove the patch because the test is green without
    it; green without it means the import is not being exercised.
    """
    import importlib

    monkeypatch.setattr(sys, "argv", [f"{MODULE_SCOPE_SCRIPT}.py"])
    module = importlib.import_module(f"analysis.paper1.{MODULE_SCOPE_SCRIPT}")
    assert getattr(module, "main", None) is None, (
        f"{MODULE_SCOPE_SCRIPT} now has a main(); the module-scope shape these "
        "tests cover may no longer exist, and they must be re-pointed or removed"
    )


def test_a_module_scope_script_writes_where_it_was_told(run_figure, tmp_path):
    """sys.argv must be set before the import, not around a main() call."""
    out = tmp_path / "asked_for"
    out.mkdir()

    status = run_figure(MODULE_SCOPE_SCRIPT, _argv(out))

    assert status == 0, f"runner reported {status}"
    assert (out / EXPECTED).is_file(), (
        f"{EXPECTED} was not written into the directory passed to the runner. "
        "The script parses sys.argv at import, so an argv patched after the "
        "import leaves it writing to its own default instead."
    )


def test_running_the_same_script_twice_writes_twice(run_figure, tmp_path):
    """A cached module must be re-executed, or the second call is a no-op."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()

    run_figure(MODULE_SCOPE_SCRIPT, _argv(first))
    run_figure(MODULE_SCOPE_SCRIPT, _argv(second))

    assert (first / EXPECTED).is_file(), "the first call produced nothing"
    assert (second / EXPECTED).is_file(), (
        "the second call produced nothing. Python imports a module once per "
        "process and this script does its work at import, so without an "
        "explicit reload a repeat call returns 0 having written no figure."
    )
