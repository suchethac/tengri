# SPDX-License-Identifier: BSD-3-Clause
"""Regression: DL07 lazy loader must fail loudly, and find data/ from any CWD.

Two coupled bugs (found rendering the Prospector reproduction notebook from
its own directory):

1. ``_DATA_CANDIDATES[0]`` used ``parents[4]`` — the repo root before this
   module moved into the ``emission/`` subpackage, ``src/data`` after — so
   template discovery silently depended on ``Path("data")`` (CWD-relative)
   and broke for any process not started at the repo root.
2. ``_dl07_lazy_wrapper`` marked the model resolved *before* the file check;
   after one failed probe (exception swallowed upstream) every later call
   re-dispatched to itself — ``RecursionError`` instead of the intended
   ``FileNotFoundError``. The factory-made wrappers had a loud-failure
   guard; the hand-written DL07 one did not.
"""

import os
import pathlib

import pytest

from tengri.components.dust.emission import emission as E

pytestmark = pytest.mark.regression_bug


def test_repo_root_candidate_points_at_repo_data():
    """The first static candidate must be <repo>/data under the src layout."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    assert E._DATA_CANDIDATES[0] == repo_root / "data"


def test_find_data_file_is_cwd_independent(tmp_path, monkeypatch):
    """A template visible in <repo>/data must resolve from any CWD."""
    probe = "_test_bug_dl07_probe.h5"
    repo_data = pathlib.Path(__file__).resolve().parents[3] / "data"
    if not repo_data.is_dir():
        pytest.skip("no data/ directory in this checkout")
    probe_path = repo_data / probe
    probe_path.write_bytes(b"x")
    try:
        monkeypatch.chdir(tmp_path)  # a CWD with no data/ anywhere above it
        assert E._find_data_file(probe) == str(probe_path)
    finally:
        os.unlink(probe_path)


def test_dl07_poisoned_state_raises_not_recurses(monkeypatch):
    """A failed first resolution must raise RuntimeError, not recurse."""
    monkeypatch.setitem(E.DUST_EMISSION_MODELS, "draine_li2007", E._dl07_lazy_wrapper)
    monkeypatch.setattr(E, "_resolved", {"draine_li2007"})
    with pytest.raises(RuntimeError, match="inconsistent state"):
        E._dl07_lazy_wrapper()
