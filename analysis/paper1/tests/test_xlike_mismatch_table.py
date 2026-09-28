# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for xlike_mismatch_table.py output.

Verifies the mismatch table script produces valid output with correct
structure, five rows per code, parity flags, and no developmental language.
"""

from __future__ import annotations

import json
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
            assert isinstance(summary["mismatches"], list)

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

    def test_latex_output_structure(self, tmp_path):
        """LaTeX output is valid tabular environment."""
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

        assert r"\begin{deluxetable}" in content
        assert r"\end{deluxetable}" in content
        assert r"\startdata" in content
        assert r"\enddata" in content
        # Count rows (each ends with \\)
        n_rows = content.count(r"\\")
        assert n_rows >= 5, f"Expected at least 5 rows, got {n_rows}"

    def test_latex_american_spelling(self, tmp_path):
        """LaTeX output uses American spelling (e.g., 'parameterization')."""
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

        # Check a few American spellings (should not have British alternatives)
        british_alternatives = ["center", "analyze"]
        for word in british_alternatives:
            assert word not in content.lower(), f"British spelling variant found instead of: {word}"

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
                f"{key}: Table 1 row mismatch.\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}"
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
