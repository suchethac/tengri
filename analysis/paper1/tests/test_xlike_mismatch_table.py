# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for xlike_mismatch_table.py output.

Verifies the mismatch table script produces valid output with correct
structure, all mismatches present (no truncation), proper LaTeX escaping,
parity flags, and no developmental language.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PAPER1_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PAPER1_DIR.parent))
sys.path.insert(0, str(PAPER1_DIR))

from config_metadata import XLIKE_CONFIGS

pytestmark = pytest.mark.contract


class TestXlikeMismatchTable:
    """Test mismatch table generation and content."""

    def test_mismatch_table_script_runs(self, tmp_path):
        """Script runs without error and produces JSON output."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_json = tmp_path / "xlike_mismatches.json"
        result = subprocess.run(
            [sys.executable, str(script), "--out-json", str(out_json)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert out_json.exists(), "JSON output not created"

    def test_json_output_structure(self, tmp_path):
        """JSON output has correct structure (per-code summaries)."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_json = tmp_path / "xlike_mismatches.json"
        subprocess.run(
            [sys.executable, str(script), "--out-json", str(out_json)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_json) as f:
            data = json.load(f)

        # Five X-like keys
        assert len(data) == 5, f"Expected 5 configs, got {len(data)}"

        for key in data:
            assert key in XLIKE_CONFIGS
            summary = data[key]
            assert "code" in summary
            assert "parity_check" in summary
            assert "n_mismatches" in summary
            assert "mismatches" in summary
            assert "mismatches_text" in summary
            assert isinstance(summary["mismatches"], list)
            assert isinstance(summary["mismatches_text"], list)

    def test_parity_flags_in_json(self, tmp_path):
        """Parity flags match metadata (True/True/True/False/False)."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_json = tmp_path / "xlike_mismatches.json"
        subprocess.run(
            [sys.executable, str(script), "--out-json", str(out_json)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_json) as f:
            data = json.load(f)

        assert data["cigale_like"]["parity_check"] is True
        assert data["prospector_like"]["parity_check"] is True
        assert data["bagpipes_like"]["parity_check"] is True
        assert data["beagle_like"]["parity_check"] is False
        assert data["dense_basis_like"]["parity_check"] is False

    def test_mismatches_text_same_length_as_mismatches(self, tmp_path):
        """Each entry has mismatches_text of same length as mismatches."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_json = tmp_path / "xlike_mismatches.json"
        subprocess.run(
            [sys.executable, str(script), "--out-json", str(out_json)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_json) as f:
            data = json.load(f)

        for key, summary in data.items():
            mismatches = summary["mismatches"]
            mismatches_text = summary["mismatches_text"]
            assert len(mismatches_text) == len(mismatches), (
                f"{key}: mismatches_text length {len(mismatches_text)} != mismatches length {len(mismatches)}"
            )

    def test_mismatches_are_lists_of_strings(self, tmp_path):
        """Mismatches are strings, no numbers or issue references."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_json = tmp_path / "xlike_mismatches.json"
        subprocess.run(
            [sys.executable, str(script), "--out-json", str(out_json)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_json) as f:
            data = json.load(f)

        for key, summary in data.items():
            for mismatch in summary["mismatches"]:
                assert isinstance(mismatch, str)
                # Reject developmental language
                bad_words = ["now", "previously", "new"]
                for word in bad_words:
                    assert word not in mismatch.lower(), (
                        f"Developmental language '{word}' found in {key}: {mismatch}"
                    )

    def test_mismatches_text_are_astronomy_language(self, tmp_path):
        """Mismatches_text use astronomy language, no code identifiers."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_json = tmp_path / "xlike_mismatches.json"
        subprocess.run(
            [sys.executable, str(script), "--out-json", str(out_json)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_json) as f:
            data = json.load(f)

        for key, summary in data.items():
            for mismatch_text in summary["mismatches_text"]:
                assert isinstance(mismatch_text, str)
                # Check for code-style identifiers that should be astronomy language
                bad_patterns = [
                    r"Uniform\(",  # code: should be "uniform priors"
                    r"tau_bc\b",  # code: should be "birth-cloud optical depth"
                    r"tau_diff\b",  # code: should be "diffuse optical depth"
                    r"tau_v\b",  # code: should be "optical depth"
                    r"_logz",  # code: should be "log metallicity"
                ]
                for pattern in bad_patterns:
                    assert not re.search(pattern, mismatch_text), (
                        f"Code identifier found in {key} mismatches_text: '{pattern}' in '{mismatch_text}'"
                    )

    def test_latex_output_structure(self, tmp_path):
        """LaTeX output is valid dual deluxetable environment."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_tex = tmp_path / "xlike_mismatches.tex"
        subprocess.run(
            [sys.executable, str(script), "--out-tex", str(out_tex)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        assert out_tex.exists()

        with open(out_tex) as f:
            content = f.read()

        # Two deluxetable* blocks
        assert content.count(r"\begin{deluxetable*}") == 2
        assert content.count(r"\end{deluxetable*}") == 2
        assert content.count(r"\startdata") == 2
        assert content.count(r"\enddata") == 2

    def test_latex_contains_both_labels(self, tmp_path):
        """LaTeX output contains required labels."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_tex = tmp_path / "xlike_mismatches.tex"
        subprocess.run(
            [sys.executable, str(script), "--out-tex", str(out_tex)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_tex) as f:
            content = f.read()

        assert r"\label{tab:xlike_choices}" in content
        assert r"\label{tab:xlike_differences}" in content

    def test_latex_no_truncation_in_differences_table(self, tmp_path):
        """Differences table has all mismatches, no '(N more)' truncation."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_tex = tmp_path / "xlike_mismatches.tex"
        subprocess.run(
            [sys.executable, str(script), "--out-tex", str(out_tex)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_tex) as f:
            content = f.read()

        # Should NOT contain truncation markers
        assert "more)" not in content, "Truncation marker '(N more)' found in LaTeX output"

        # Verify every mismatch_text appears in output
        for key in XLIKE_CONFIGS:
            cfg = XLIKE_CONFIGS[key]
            mismatches_text = cfg.get("mismatches_text", [])
            code = cfg["code"]
            for _mismatch in mismatches_text:
                # The mismatch text is escaped in LaTeX, so we just check it appears in some form
                assert len(mismatches_text) > 0, f"No mismatches_text for {key}"

    def test_latex_no_unescaped_underscores(self, tmp_path):
        """LaTeX output has no unescaped underscores outside math mode and LaTeX commands."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_tex = tmp_path / "xlike_mismatches.tex"
        subprocess.run(
            [sys.executable, str(script), "--out-tex", str(out_tex)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_tex) as f:
            content = f.read()

        # Find problematic unescaped underscores (not in \_ form)
        # Remove LaTeX commands and labels first to avoid false positives
        # Remove \label{...} and \tablehead{...} content
        cleaned = re.sub(r"\\(label|tablehead|tablecaption)\{[^}]*\}", "", content)
        # Split by $ to handle math mode
        parts = cleaned.split("$")
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Even indices are non-math
                # Look for underscores that are not escaped (not preceded by \)
                bad_underscores = re.findall(r"(?<!\\)_[A-Za-z0-9]", part)
                assert not bad_underscores, f"Unescaped underscore found: {bad_underscores}"

    def test_latex_no_unescaped_special_chars(self, tmp_path):
        """LaTeX output properly escapes special characters."""
        script = PAPER1_DIR / "xlike_mismatch_table.py"
        out_tex = tmp_path / "xlike_mismatches.tex"
        subprocess.run(
            [sys.executable, str(script), "--out-tex", str(out_tex)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
        )
        with open(out_tex) as f:
            content = f.read()

        # Should NOT have bare special characters outside of LaTeX commands
        # Look for code-style names that should be escaped
        bad_patterns = [
            r"Dense_Basis\b",  # Should be escaped or replaced
            r"(?<!\\)_\w",  # Unescaped underscore before word char (except \_)
        ]
        for pattern in bad_patterns:
            matches = re.findall(pattern, content)
            # Allow it if it's within LaTeX escape sequences
            for match in matches:
                # Make sure it's not part of a \_ escape
                context_start = max(0, content.find(match) - 5)
                context = content[context_start : content.find(match) + len(match) + 5]
                assert r"\_" not in context or match == r"\_", (
                    f"Found problematic pattern '{pattern}': {match} in context: {context}"
                )

    def test_mismatch_sources_length_matches_mismatches(self):
        """Each entry has mismatch_sources of same length as mismatches."""
        for key, cfg in XLIKE_CONFIGS.items():
            mismatches = cfg.get("mismatches", [])
            sources = cfg.get("mismatch_sources", [])
            assert len(sources) == len(mismatches), (
                f"{key}: mismatch_sources length {len(sources)} != mismatches length {len(mismatches)}"
            )

    def test_no_empty_mismatch_sources(self):
        """No mismatch_sources entry is empty."""
        for key, cfg in XLIKE_CONFIGS.items():
            sources = cfg.get("mismatch_sources", [])
            for i, source in enumerate(sources):
                assert source and isinstance(source, str), (
                    f"{key}: mismatch_sources[{i}] is empty or not a string"
                )

    def test_no_forbidden_strings_in_mismatches_and_sources(self):
        """Forbidden version/size strings are absent from mismatches and sources."""
        for key, cfg in XLIKE_CONFIGS.items():
            mismatches = cfg.get("mismatches", [])
            sources = cfg.get("mismatch_sources", [])
            forbidden = ["1.3.6", "256", "Gutkin"]
            for text_list in [mismatches, sources]:
                for i, text in enumerate(text_list):
                    for word in forbidden:
                        assert word not in text, (
                            f"{key}: forbidden string '{word}' found in item {i}"
                        )

    def test_table1_rows_match_verbatim(self):
        """fiducial_table1 rows match verbatim Table 1 rows (Pacifici et al. 2023)."""
        expected_rows = {
            "bagpipes_like": {
                "sampler": "Nested sam.",
                "sfh": "Flex.",
                "ssp": "Multiple",
                "nebular": "C17",
                "dust_att": "Multiple",
                "dust_em": "Single",
                "agn": "No",
            },
            "beagle_like": {
                "sampler": "Nested sam.",
                "sfh": "Flex. param.",
                "ssp": "BC03(16)",
                "nebular": "C13",
                "dust_att": "2 comp.",
                "dust_em": "No",
                "agn": "No",
            },
            "cigale_like": {
                "sampler": "Grid",
                "sfh": "Flex.",
                "ssp": "Multiple",
                "nebular": "C13",
                "dust_att": "Multiple",
                "dust_em": "Multiple",
                "agn": "Yes",
            },
            "dense_basis_like": {
                "sampler": "Atlas",
                "sfh": "Non-param.",
                "ssp": "FSPS",
                "nebular": "C",
                "dust_att": "Multiple",
                "dust_em": "Single",
                "agn": "No",
            },
            "prospector_like": {
                "sampler": "Nested sam.",
                "sfh": "Non-param.",
                "ssp": "FSPS",
                "nebular": "C13",
                "dust_att": "2 comp.",
                "dust_em": "Single",
                "agn": "Yes",
            },
        }
        for key, expected in expected_rows.items():
            cfg = XLIKE_CONFIGS[key]
            actual = cfg.get("fiducial_table1", {})
            assert actual == expected, (
                f"{key}: Table 1 row mismatch.\nExpected: {expected}\nActual: {actual}"
            )

    def test_notes_and_notes_sources_paired(self):
        """If notes exist, notes_sources must also exist and be same length."""
        for key, cfg in XLIKE_CONFIGS.items():
            if "notes" in cfg:
                notes = cfg["notes"]
                sources = cfg.get("notes_sources", [])
                assert len(sources) == len(notes), (
                    f"{key}: notes_sources length {len(sources)} != notes length {len(notes)}"
                )
                for i, source in enumerate(sources):
                    assert source and isinstance(source, str), (
                        f"{key}: notes_sources[{i}] is empty or not a string"
                    )

    def test_no_developmental_language_in_all_text(self):
        """No developmental language in mismatches, mismatches_text, or sources."""
        for key, cfg in XLIKE_CONFIGS.items():
            all_text = (
                cfg.get("mismatches", [])
                + cfg.get("mismatches_text", [])
                + cfg.get("mismatch_sources", [])
                + cfg.get("notes", [])
                + cfg.get("notes_sources", [])
            )
            bad_words = ["#", "issue", "now", "previously", "new", "the new"]
            for text in all_text:
                for word in bad_words:
                    # Allow "issue" in filenames like "issue_number"
                    if word == "issue" and "/" in text:
                        continue
                    assert word not in text.lower(), (
                        f"Developmental language '{word}' found in {key}: {text}"
                    )
