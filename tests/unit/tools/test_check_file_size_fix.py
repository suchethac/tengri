# SPDX-License-Identifier: BSD-3-Clause
"""The file-size ratchet's --fix mode: mechanical bookkeeping, deliberate additions.

Three merges in one week shipped with red ratchet pins (#1167, #1204, #1188)
because pin updates were manual. --fix makes them a one-command habit; the
--allow-new gate keeps NEW >800-line files a deliberate decision.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "check_file_size.py"


def _load_tool(tmp_path, allowlist):
    spec = importlib.util.spec_from_file_location("check_file_size_uut", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_file_size_uut"] = mod
    spec.loader.exec_module(mod)
    mod.REPO_ROOT = tmp_path
    mod.ALLOWLIST_PATH = tmp_path / "allowlist.json"
    mod.MAX_LINES = 5
    (tmp_path / "allowlist.json").write_text(json.dumps(allowlist))
    (tmp_path / "src").mkdir()
    return mod


def _write(tmp_path, name, n_lines):
    (tmp_path / "src" / name).write_text("\n".join(f"# {i}" for i in range(n_lines)) + "\n")


def test_fix_updates_grown_pin_and_removes_shrunk_and_orphan(tmp_path):
    mod = _load_tool(
        tmp_path,
        {"src/grown.py": 8, "src/shrunk.py": 9, "src/gone.py": 12},
    )
    _write(tmp_path, "grown.py", 10)  # grew past its pin of 8
    _write(tmp_path, "shrunk.py", 3)  # now under MAX_LINES
    # gone.py never written -> orphaned entry
    assert mod.main([]) == 1  # gate mode still fails
    assert mod.main(["--fix"]) == 0
    out = json.loads((tmp_path / "allowlist.json").read_text())
    assert out == {"src/grown.py": 10}
    assert mod.main([]) == 0  # gate green after fix


def test_fix_refuses_new_files_without_allow_new(tmp_path):
    mod = _load_tool(tmp_path, {})
    _write(tmp_path, "big_new.py", 9)
    assert mod.main(["--fix"]) == 1  # refused: deliberate decision required
    assert json.loads((tmp_path / "allowlist.json").read_text()) == {}
    assert mod.main(["--fix", "--allow-new"]) == 0
    out = json.loads((tmp_path / "allowlist.json").read_text())
    assert out == {"src/big_new.py": 9}


def test_unknown_argument_rejected(tmp_path):
    mod = _load_tool(tmp_path, {})
    assert mod.main(["--fxi"]) == 2
