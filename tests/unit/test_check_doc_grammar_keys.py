# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the doc grammar keys guard."""

import pathlib
import subprocess
import sys
import tempfile

import pytest


def test_check_doc_grammar_keys_guard() -> None:
    """Run the guard on the live repository and ensure it passes.

    This test imports and runs tools/check_doc_grammar_keys.py::main()
    to verify that all grammar structural keys are properly documented
    and that documentation doesn't refer to non-existent keys.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    guard_script = repo_root / "tools" / "check_doc_grammar_keys.py"

    assert guard_script.exists(), f"Guard script not found at {guard_script}"

    # Run the guard as a subprocess to avoid import order issues
    result = subprocess.run(
        [sys.executable, str(guard_script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        pytest.fail(f"Grammar documentation guard failed:\n{result.stdout}\n{result.stderr}")

    # If we got here, the guard passed
    assert result.returncode == 0
    assert "✓" in result.stdout or "complete" in result.stdout.lower()


def test_colon_separator_format() -> None:
    """Test that structural keys using colon separator are parsed correctly.

    Verifies that a bullet line using ': ' (colon-space) instead of '—'
    (em-dash) correctly extracts only the key name, not the type values
    that follow the separator.

    Example bullet format:
    - `'type'`: Selects the model family; values include `'dpl'`, `'delayed_tau'`
    """
    # Import here to avoid issues if the module can't be imported at collection time
    from tools.check_doc_grammar_keys import _read_doc_keys_from_file

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        # Create a minimal doc file with colon-separated bullets
        doc_content = """\
### Star-formation history: `sfh`

**Structural keys:**
- `'type'`: Selects the SFH family; values are `'dpl'`, `'delayed_tau'`
- `'all_params'`: Wildcard: FREE or Fixed. Exact synonym: `'other_params'`
- `'age_kernel'`: Integration method; default is `'cic'`

### Metallicity: `met`

**Structural keys:**
- `'type'`: Metallicity model; values are `'table'`, `'ramp'`, etc.
"""
        doc_file = tmpdir_path / "model_configuration.md"
        doc_file.write_text(doc_content, encoding="utf-8")

        # Parse the documentation
        doc_keys = _read_doc_keys_from_file(doc_file)

        # Verify that only the keys are extracted, not the type values.
        # The all_params bullet includes other_params via the "Exact synonym:" callout.
        assert doc_keys["sfh"] == {"type", "all_params", "other_params", "age_kernel"}
        assert doc_keys["met"] == {"type"}

        # Ensure that type values like 'dpl', 'delayed_tau', 'table', 'ramp'
        # are not mistakenly extracted as keys
        assert "dpl" not in doc_keys["sfh"]
        assert "delayed_tau" not in doc_keys["sfh"]
        assert "cic" not in doc_keys["sfh"]
        assert "table" not in doc_keys["met"]
        assert "ramp" not in doc_keys["met"]
