# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/check_untracked_data_gates.py (#3448).

A test gated on a data file that CI does not have skips on every runner and says
nothing. The guard finds those gates by AST and holds them in a shrink-only
baseline. These tests pin the scanner's rules and the baseline comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

import check_untracked_data_gates as guard

pytestmark = pytest.mark.contract


def _gated(source: str) -> set[str]:
    return guard.collect_gates(source)


def test_gate_on_untracked_name_is_reported():
    source = (
        'import pytest\n@pytest.mark.skipif(find_data_str("x.h5") is None)\ndef test_a(): pass\n'
    )
    found = guard.find_untracked({"tests/test_a.py": source}, tracked=set(), workflow_text="")
    assert found == {("tests/test_a.py", "x.h5")}


def test_same_name_tracked_in_git_is_not_reported():
    source = 'skipif(find_data_str("x.h5") is None)\n'
    found = guard.find_untracked({"tests/test_a.py": source}, tracked={"x.h5"}, workflow_text="")
    assert found == set()


def test_name_named_by_a_workflow_download_is_not_reported():
    source = 'skipif(find_data_str("x.h5") is None)\n'
    found = guard.find_untracked(
        {"tests/test_a.py": source}, tracked=set(), workflow_text="curl .../x.h5"
    )
    assert found == set()


def test_stale_baseline_line_fails_as_stale():
    recorded = {("tests/test_a.py", "x.h5")}
    problems = guard.compare(found=set(), recorded=recorded)
    assert problems == ["  stale: tests/test_a.py  x.h5"]


def test_new_finding_outside_baseline_fails():
    problems = guard.compare(found={("tests/test_b.py", "y.npz")}, recorded=set())
    assert problems == ["  untracked gate  tests/test_b.py  y.npz"]


def test_non_data_strings_in_skip_arguments_are_ignored():
    source = (
        'skipif(not ok, reason="data/x.h5 not found")\n'
        'pytest.skip("no data here")\n'
        'skipif(find_data_str("a name with spaces.h5") is None)\n'
    )
    assert _gated(source) == set()


def test_name_the_module_itself_writes_is_not_a_gate():
    source = '(tmp / "probe.npz").write_bytes(b"")\nskipif(find_data_str("probe.npz") is None)\n'
    assert _gated(source) == set()


def test_lookup_under_assert_is_not_a_gate():
    source = 'assert find_data_str("__absent__.h5") is None\n'
    assert _gated(source) == set()


def test_module_constant_named_by_a_gate_is_resolved():
    source = '_GRID = "data/grid.h5"\nskipif(not os.path.exists(_GRID))\n'
    assert _gated(source) == {"grid.h5"}


def test_scanner_walks_tests_tree(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_gated.py").write_text('skipif(find_data_str("z.h5") is None)\n')
    sources = guard._read_sources(tmp_path)
    assert list(sources) == ["tests/test_gated.py"]
    assert guard.gate_table(sources) == {"z.h5": {"tests/test_gated.py"}}


def test_path_join_names_a_gate_without_a_gate_call():
    source = 'from pathlib import Path\nGRID = Path("data") / "synth" / "grid_agn.hdf5"\n'
    assert guard.collect_gates(source) == {"grid_agn.hdf5"}


def test_path_join_rooted_in_a_scratch_directory_is_not_a_gate():
    source = 'def test_x(tmp_path):\n    out = tmp_path / "made_here.h5"\n'
    assert guard.collect_gates(source) == set()


def test_stand_ins_written_before_collection_count_as_available():
    conftest = 'def pytest_configure(config):\n    path = ROOT / "data" / "stand_in.h5"\n'
    assert guard.synthesized_names(conftest) == {"stand_in.h5"}
    assert guard.synthesized_names('def other():\n    p = ROOT / "x.h5"\n') == set()
