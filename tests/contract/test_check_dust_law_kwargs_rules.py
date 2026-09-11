# SPDX-License-Identifier: BSD-3-Clause
"""Tests for rule 2b (subscript dicts) and rule 4 (spelling) in tools/check_dust_law_kwargs.py.

Rule 2b: Subscript-built dicts that carry shape parameters must be allowlisted.
Rule 4: Every law call site must use the correct spelling for parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from check_dust_law_kwargs import (
    build_law_params,
    main,
    scan_call_spellings,
    scan_subscript_assignments,
)

pytestmark = pytest.mark.contract


class TestRule2b_SubscriptAssignments:
    """Rule 2b: detect subscript-built dicts with shape parameter keys."""

    def test_subscript_assignment_with_dust_slope_is_reported(self, tmp_path):
        """A dict built via subscript with dust_slope key should be flagged."""
        src_file = tmp_path / "test.py"
        src_file.write_text('d = {}\nd["dust_slope"] = -0.7\n')
        found = scan_subscript_assignments(src_file)
        assert found, f"Expected to find dust_slope assignment but got {found}"
        linenos = {lineno for lineno, key in found}
        assert 2 in linenos, "Expected line 2 to be flagged"

    def test_subscript_assignment_without_shape_kwargs_is_not_reported(self, tmp_path):
        """A dict built via subscript without shape kwargs should not be flagged."""
        src_file = tmp_path / "test.py"
        src_file.write_text('d = {}\nd["law_bc"] = "calzetti"\n')
        found = scan_subscript_assignments(src_file)
        assert not found, f"Expected no violations but got {found}"

    def test_multiple_subscript_assignments(self, tmp_path):
        """Multiple subscript assignments with shape kwargs should all be flagged."""
        src_file = tmp_path / "test.py"
        src_file.write_text(
            'd = {}\nd["dust_slope"] = -0.7\nd["dust_Rv"] = 3.1\nd["law"] = "calzetti"\n'
        )
        found = scan_subscript_assignments(src_file)
        keys_found = {key for lineno, key in found}
        assert "dust_slope" in keys_found
        assert "dust_Rv" in keys_found
        assert "law" not in keys_found


class TestRule4_CallSpellings:
    """Rule 4: every law call uses correct parameter spellings."""

    def test_correct_spelling_is_clean(self, tmp_path):
        """A law call with correct spelling should not be flagged."""
        src_file = tmp_path / "test.py"
        src_file.write_text("power_law(wavelength, dust_slope=-0.7)\n")
        law_params = {"power_law": {"wavelength", "dust_slope"}}
        found = scan_call_spellings(src_file, law_params)
        assert not found, f"Expected no violations but got {found}"

    def test_deprecated_spelling_is_reported(self, tmp_path):
        """A law call with deprecated n_slope should be flagged."""
        src_file = tmp_path / "test.py"
        src_file.write_text("power_law(wavelength, n_slope=-0.7)\n")
        law_params = {"power_law": {"wavelength", "dust_slope"}}
        found = scan_call_spellings(src_file, law_params)
        assert found, f"Expected to find n_slope but got {found}"
        reported = {(lineno, callee, kw) for lineno, callee, kw in found}
        assert any(kw == "n_slope" for _, _, kw in reported)

    def test_unknown_keyword_is_reported(self, tmp_path):
        """A law call with unknown keyword should be flagged."""
        src_file = tmp_path / "test.py"
        src_file.write_text("power_law(wavelength, Rv=3.1)\n")
        law_params = {"power_law": {"wavelength", "dust_slope"}}
        found = scan_call_spellings(src_file, law_params)
        assert found, f"Expected to find Rv but got {found}"
        reported = {kw for _, _, kw in found}
        assert "Rv" in reported

    def test_splat_kwargs_are_clean(self, tmp_path):
        """A law call with **kw splat should not be flagged."""
        src_file = tmp_path / "test.py"
        src_file.write_text("power_law(wavelength, **kw)\n")
        law_params = {"power_law": {"wavelength", "dust_slope"}}
        found = scan_call_spellings(src_file, law_params)
        assert not found, f"Expected **kw to be clean but got {found}"

    def test_multiple_violations_in_one_call(self, tmp_path):
        """Multiple bad keywords in one call should all be reported."""
        src_file = tmp_path / "test.py"
        src_file.write_text("power_law(wavelength, n_slope=-0.7, Rv=3.1)\n")
        law_params = {"power_law": {"wavelength", "dust_slope"}}
        found = scan_call_spellings(src_file, law_params)
        keywords = {kw for _, _, kw in found}
        assert keywords >= {"n_slope", "Rv"}

    def test_call_inside_pytest_raises_is_excluded(self, tmp_path):
        """A call inside pytest.raises(...) should be excluded."""
        src_file = tmp_path / "test.py"
        src_file.write_text(
            "import pytest\n"
            "with pytest.raises(TypeError):\n"
            "    power_law(wavelength, n_slope=-0.7)\n"
        )
        law_params = {"power_law": {"wavelength", "dust_slope"}}
        found = scan_call_spellings(src_file, law_params)
        assert not found, f"Expected raises block to be excluded but got {found}"

    def test_call_outside_pytest_raises_is_reported(self, tmp_path):
        """Same call outside raises block should be reported."""
        src_file = tmp_path / "test.py"
        src_file.write_text(
            "import pytest\n"
            "power_law(wavelength, n_slope=-0.7)\n"
            "with pytest.raises(TypeError):\n"
            "    pass\n"
        )
        law_params = {"power_law": {"wavelength", "dust_slope"}}
        found = scan_call_spellings(src_file, law_params)
        assert found, f"Expected to find n_slope outside raises but got {found}"


class TestLawParamsTable:
    """Build the law_params table from real attenuation.py."""

    @staticmethod
    def _attenuation_path():
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / "src" / "tengri" / "components" / "dust" / "attenuation.py"

    def test_law_params_contains_power_law(self):
        """power_law should be in the table with dust_slope."""
        law_params = build_law_params(self._attenuation_path())
        assert "power_law" in law_params
        assert "dust_slope" in law_params["power_law"]

    def test_law_params_does_not_contain_deprecated_spelling(self):
        """law_params should never have n_slope (deprecated)."""
        law_params = build_law_params(self._attenuation_path())
        for params in law_params.values():
            assert "n_slope" not in params, f"Found deprecated n_slope in {params}"

    def test_law_params_contains_cardelli_with_dust_Rv(self):
        """cardelli should accept dust_Rv."""
        law_params = build_law_params(self._attenuation_path())
        assert "cardelli" in law_params
        assert "dust_Rv" in law_params["cardelli"]

    def test_law_params_always_includes_wavelength(self):
        """Every law should have wavelength as a parameter."""
        law_params = build_law_params(self._attenuation_path())
        for law_name, params in law_params.items():
            assert "wavelength" in params, f"{law_name} missing wavelength"


class TestEndToEnd:
    """End-to-end: main() runs on the real repository."""

    def test_main_is_clean_on_the_repository(self, capsys):
        """main() exits cleanly (0) on the real repository.

        Negative tests inside pytest.raises(...) blocks are excluded from rule 4,
        so the repository should pass all checks.
        """
        exit_code = main()
        captured = capsys.readouterr()

        # No violations: main() returns 0
        if exit_code != 0:
            print("CAPTURED STDERR:", captured.err, file=sys.stderr)
        assert exit_code == 0, f"Expected main() to return 0 but got {exit_code}"
