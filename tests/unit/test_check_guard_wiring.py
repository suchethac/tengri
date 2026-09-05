# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for check_guard_wiring.py.

Tests the guard wiring census: all check_*.py files must be wired into CI or
explicitly declared not wired in their module docstring (#2050).
"""

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_guard_wiring.py"
_spec = importlib.util.spec_from_file_location("check_guard_wiring", _TOOL)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class TestDocstringDetection:
    """Test module docstring extraction and declaration detection."""

    def test_extract_module_docstring(self, tmp_path):
        """Extracts module-level docstring."""
        py_file = tmp_path / "test.py"
        py_file.write_text('"""This is a docstring."""\n')
        result = mod.extract_module_docstring(py_file)
        assert result == "This is a docstring."

    def test_extract_docstring_with_body(self, tmp_path):
        """Docstring followed by code."""
        py_file = tmp_path / "test.py"
        py_file.write_text('"""Module doc."""\n\ndef foo():\n    pass\n')
        result = mod.extract_module_docstring(py_file)
        assert result == "Module doc."

    def test_no_docstring(self, tmp_path):
        """File with no docstring."""
        py_file = tmp_path / "test.py"
        py_file.write_text("def foo():\n    pass\n")
        result = mod.extract_module_docstring(py_file)
        assert result is None

    def test_is_explicitly_not_wired_with_declaration(self):
        """Docstring with 'CI: not wired' declaration is detected."""
        docstring = "Some doc.\nCI: not wired — backlog too large.\nMore doc."
        assert mod.is_explicitly_not_wired(docstring) is True

    def test_is_explicitly_not_wired_with_em_dash(self):
        """Works with em-dash."""
        docstring = "CI: not wired — reason"
        assert mod.is_explicitly_not_wired(docstring) is True

    def test_is_explicitly_not_wired_with_plain_hyphen(self):
        """Works with plain hyphen."""
        docstring = "CI: not wired - reason"
        assert mod.is_explicitly_not_wired(docstring) is True

    def test_is_explicitly_not_wired_empty_reason_rejected(self):
        """Empty reason is rejected."""
        docstring = "CI: not wired — "
        assert mod.is_explicitly_not_wired(docstring) is False

    def test_is_explicitly_not_wired_no_declaration(self):
        """Without the declaration, returns False."""
        docstring = "Some doc.\nNot wired because reasons.\nMore doc."
        assert mod.is_explicitly_not_wired(docstring) is False


class TestWiringCheck:
    """Test the main wiring check against the real repo."""

    def test_main_exits_zero_on_real_repo(self):
        """The real repo should pass (all guards wired or declared)."""
        result = mod.main()
        assert result == 0


class TestAgainstTempRepo:
    """Test guard detection with a temporary repo structure."""

    def test_main_fails_when_guard_unwired_and_undeclared(self, tmp_path, monkeypatch):
        """Guard that is neither wired nor declared fails."""
        # Create fake tools dir with an unwired guard (only one guard)
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        unwired_guard = tools_dir / "check_unwired_example.py"
        unwired_guard.write_text('"""A guard with no wiring.\n"""\n\ndef main():\n    pass\n')

        # Create fake workflows dir (empty)
        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        (workflows_dir / "tests.yml").write_text("")

        # Patch to use the temp directory
        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "WORKFLOWS_DIR", workflows_dir)

        result = mod.main()
        assert result == 1

    def test_main_passes_when_guard_is_wired(self, tmp_path, monkeypatch):
        """Guard wired into workflow passes."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        wired_guard = tools_dir / "check_wired_example.py"
        wired_guard.write_text('"""A wired guard.\n"""\n')

        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        (workflows_dir / "tests.yml").write_text("- run: python tools/check_wired_example.py\n")

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "WORKFLOWS_DIR", workflows_dir)

        result = mod.main()
        assert result == 0

    def test_main_passes_when_guard_is_declared_not_wired(self, tmp_path, monkeypatch):
        """Guard with explicit 'not wired' declaration passes."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        declared_guard = tools_dir / "check_declared_example.py"
        declared_guard.write_text(
            '"""A guard with explicit not-wired declaration.\n\n'
            "CI: not wired — backlog too large (#2050).\n"
            '"""\n'
        )

        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        (workflows_dir / "tests.yml").write_text("")

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "WORKFLOWS_DIR", workflows_dir)

        result = mod.main()
        assert result == 0

    def test_guard_named_only_in_comment_is_not_wired(self, tmp_path, monkeypatch):
        """Guard mentioned only in a YAML comment is NOT counted as wired."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        guard = tools_dir / "check_comment_only.py"
        guard.write_text('"""A guard mentioned only in comments.\n"""\n')

        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        # Guard name appears only in a comment, not in a run: line
        (workflows_dir / "tests.yml").write_text(
            "      # Verify check_comment_only is working\n"
            "      - run: python tools/other_guard.py\n"
        )

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "WORKFLOWS_DIR", workflows_dir)

        result = mod.main()
        assert result == 1  # Should fail: guard is unwired and undeclared

    def test_guard_with_prefix_match_not_falsely_wired(self, tmp_path, monkeypatch):
        """Guard check_foo is NOT wired just because check_foo_bar is."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        # Create two guards: check_foo and check_foo_bar
        (tools_dir / "check_foo.py").write_text('"""Guard foo.\n"""\n')
        (tools_dir / "check_foo_bar.py").write_text('"""Guard foo_bar.\n"""\n')

        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        # Only check_foo_bar is wired, not check_foo
        (workflows_dir / "tests.yml").write_text("      - run: python tools/check_foo_bar.py\n")

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "WORKFLOWS_DIR", workflows_dir)

        result = mod.main()
        # Should fail: check_foo is unwired and undeclared
        assert result == 1

    def test_block_scalar_wiring_recognized(self, tmp_path, monkeypatch):
        """Guard wired via block-scalar (python command on its own line) is recognized."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        guard = tools_dir / "check_block_scalar.py"
        guard.write_text('"""A guard wired via block scalar.\n"""\n')

        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        # Block-scalar format with python command indented on its own line
        (workflows_dir / "tests.yml").write_text(
            "- name: Check guard\n  run: |\n    python tools/check_block_scalar.py\n"
        )

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "WORKFLOWS_DIR", workflows_dir)

        result = mod.main()
        assert result == 0

    def test_python_command_in_comment_not_wired(self, tmp_path, monkeypatch):
        """Python command on a comment line does NOT count as wired."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        guard = tools_dir / "check_commented.py"
        guard.write_text('"""A guard mentioned only in a comment line.\n"""\n')

        workflows_dir = tmp_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True)
        # Guard mentioned in a comment line only
        (workflows_dir / "tests.yml").write_text(
            "- name: Check guard\n"
            "  run: |\n"
            "    # python tools/check_commented.py\n"
            "    python tools/other.py\n"
        )

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "WORKFLOWS_DIR", workflows_dir)

        result = mod.main()
        assert result == 1  # Should fail: guard is unwired and undeclared
