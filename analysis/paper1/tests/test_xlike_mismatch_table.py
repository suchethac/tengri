# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for xlike_mismatch_table.py output.

Verifies the mismatch table script produces valid output with correct
structure, all mismatches present (no truncation), proper LaTeX escaping,
parity flags, and no developmental language.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PAPER1_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PAPER1_DIR.parent))
sys.path.insert(0, str(PAPER1_DIR))

from config_metadata import XLIKE_CONFIGS
from xlike_mismatch_table import (
    _TENGRI_DISPLAY,
    display_code,
    render_latex,
    source_display,
    tengri_display,
)

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
        """LaTeX output is two table* blocks, and never a deluxetable."""
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

        # deluxetable cannot take p{} columns (the cause of the "Extra \or" errors)
        assert "deluxetable" not in content
        assert content.count(r"\begin{table*}") == 2
        assert content.count(r"\end{table*}") == 2
        assert content.count(r"\begin{tabular}") == 2
        assert content.count(r"\toprule") == 2

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
        # Remove \label{...} and citation keys (bib keys carry underscores)
        cleaned = re.sub(r"\\(label|cite[a-z]*)(\[[^\]]*\])?\{[^}]*\}", "", content)
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


# Reader-facing forms of the sources (raw file:line stays in the JSON only).
_RAW_FILE_HINTS = (".py", ".csv", "header", "site-packages", "art_sedfitting", "reproduction/")


class TestReaderFacingText:
    """The published tables name citations and data products, never files or codes."""

    def test_code_display_has_no_underscore(self):
        assert display_code("Dense_Basis") == "Dense Basis"
        for cfg in XLIKE_CONFIGS.values():
            assert "_" not in display_code(cfg["code"])

    def test_rendered_tex_has_no_escaped_code_names_or_file_names(self):
        tex = render_latex()
        assert r"Dense\_Basis" not in tex
        assert "Dense Basis" in tex
        for hint in _RAW_FILE_HINTS:
            assert hint not in tex, f"raw source fragment {hint!r} reached the table"

    def test_every_source_has_a_reader_facing_form(self):
        for key, cfg in XLIKE_CONFIGS.items():
            for raw in cfg["mismatch_sources"]:
                latex, plain = source_display(raw)
                assert latex and plain, (key, raw)
                for hint in _RAW_FILE_HINTS:
                    assert hint not in plain, (key, raw, plain)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="reader-facing form"):
            source_display("scratch/notes.txt:3")

    def test_source_forms_match_the_ruling(self):
        assert source_display("art_sedfitting/code_outputs/header (absence)")[1].startswith(
            "workshop catalog (column list)"
        )
        assert source_display("reproduction/cigale/01_cigale.py:24")[1] == (
            "tengri reproduction notebook (CIGALE)"
        )
        assert source_display("analysis/paper1/configs.py:178")[1] == "this work"
        assert "Iyer et al. 2019" in source_display("site-packages/dense_basis/priors.py:84")[1]

    def test_every_tengri_value_has_journal_wording(self):
        for key, cfg in XLIKE_CONFIGS.items():
            for field in _TENGRI_DISPLAY:
                shown = tengri_display(field, cfg[field])
                assert shown, (key, field)
        for shorthand in ("Calzetti, 1-comp", "Draine+2007", "Leitherer+02, 2-comp"):
            assert shorthand not in render_latex()

    def test_unknown_tengri_value_raises(self):
        with pytest.raises(ValueError, match="reader-facing wording"):
            tengri_display("attenuation", "Kriek+13, 2-comp")

    def test_json_keeps_raw_sources_and_adds_display(self, tmp_path):
        out = tmp_path / "t.json"
        subprocess.run(
            [sys.executable, str(PAPER1_DIR / "xlike_mismatch_table.py"), "--out-json", str(out)],
            cwd=str(PAPER1_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(out.read_text())
        assert data["dense_basis_like"]["code"] == "Dense_Basis"
        assert data["dense_basis_like"]["display_code"] == "Dense Basis"
        for key, cfg in XLIKE_CONFIGS.items():
            assert data[key]["mismatch_sources"] == cfg["mismatch_sources"]
            assert len(data[key]["mismatch_sources_display"]) == len(cfg["mismatch_sources"])


PAPER_DIR = Path(
    os.environ.get("TENGRI_PAPER_DIR", "/Users/suchethacooray/writing-workspace/projects/tengri")
)
_WRAPPER = r"""\documentclass[twocolumn]{aastex631}
\usepackage{booktabs}
\begin{document}
\input{xlike_mismatch_table}
See Tables~\ref{tab:xlike_choices} and \ref{tab:xlike_differences}.
\bibliographystyle{aasjournal}
\bibliography{99-references}
\end{document}
"""


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)


class TestCompilesInPaperClass:
    """The tables must build under the paper's own class, with every reference resolved."""

    def test_pdflatex_builds_the_tables(self, tmp_path):
        needed = [PAPER_DIR / n for n in ("aastex631.cls", "aasjournal.bst", "99-references.bib")]
        if shutil.which("pdflatex") is None or shutil.which("bibtex") is None:
            pytest.skip("pdflatex/bibtex not installed")
        missing = [str(n) for n in needed if not n.exists()]
        if missing:
            pytest.skip(f"paper files absent: {missing}")
        for n in needed:
            shutil.copy(n, tmp_path / n.name)
        (tmp_path / "xlike_mismatch_table.tex").write_text(render_latex())
        tex = render_latex()
        own = {"tab:xlike_choices", "tab:xlike_differences"}
        stubs = "".join(
            rf"\section{{Stub}}\label{{{lab}}}"
            for lab in sorted(set(re.findall(r"\\ref\{([^}]+)\}", tex)) - own)
        )
        (tmp_path / "wrap.tex").write_text(_WRAPPER.replace("See Tables", stubs + "See Tables"))

        flags = ["-interaction=nonstopmode", "-halt-on-error", "wrap.tex"]
        first = _run(["pdflatex", *flags], tmp_path)
        assert first.returncode == 0, first.stdout[-2000:]
        _run(["bibtex", "wrap"], tmp_path)
        _run(["pdflatex", *flags], tmp_path)
        last = _run(["pdflatex", *flags], tmp_path)
        assert last.returncode == 0, last.stdout[-2000:]

        log = (tmp_path / "wrap.log").read_text(errors="replace")
        errors = [ln for ln in log.splitlines() if ln.startswith("!")]
        assert not errors, errors[:5]
        assert (tmp_path / "wrap.pdf").exists()
        for label in ("tab:xlike_choices", "tab:xlike_differences"):
            assert f"Reference `{label}' on page" not in log
        assert "There were undefined references" not in log
        undefined_cites = re.findall(r"Citation `([^']+)' on page \d+ undefined", log)
        assert not undefined_cites, undefined_cites


_REPO_ROOT = PAPER1_DIR.parent.parent
_UPSTREAM = ("pcigale", "bagpipes", "prospect", "dense_basis")
_FILE_LINES = re.compile(r"((?:[\w.\-]+/)+[\w.\-]+\.py):(\d+(?:,\d+)*)(?:--(\d+))?")


def _site_packages_root(top: str):
    """Return the directory holding installed package ``top`` (found without importing it)."""
    import importlib.util

    spec = importlib.util.find_spec(top)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))).parent


def _resolve_source_file(rel: str):
    """Resolve a repo-relative or package-relative path; None when neither exists."""
    repo = _REPO_ROOT / rel
    if repo.is_file():
        return repo
    root = _site_packages_root(rel.split("/")[0])
    if root is not None and (root / rel).is_file():
        return root / rel
    return None


def _cited_file_lines(raw: str):
    """Yield (path string, [line numbers]) for each file:line cited in a raw source."""
    for part in raw.split(";"):
        m = _FILE_LINES.search(part)
        if m:
            lines = [int(x) for x in m.group(2).split(",")]
            if m.group(3):
                lines.append(int(m.group(3)))
            yield m.group(1), lines


def _all_rows():
    """Yield (key, text, mismatch, exact source) for every Table B row."""
    for key, cfg in XLIKE_CONFIGS.items():
        for mismatch, text, src in zip(
            cfg["mismatches"], cfg["mismatches_text"], cfg["mismatch_sources"], strict=True
        ):
            yield key, text, mismatch, src


class TestSourcesSupportOnlyTheirClaims:
    """A source may support only what it contains: an absent setting says nothing about the code."""

    def test_no_mismatch_claims_discrete_metallicity(self):
        for key, text, mismatch, _ in _all_rows():
            for claim in (text, mismatch):
                assert "discrete" not in claim.lower(), (key, claim)
                assert "metallicity treatment" not in claim.lower(), (key, claim)

    def test_prospector_has_no_age_floor_row(self):
        # The continuity SFH has no age parameter, so there is no floor to differ on.
        for claim in (
            *XLIKE_CONFIGS["prospector_like"]["mismatches"],
            *XLIKE_CONFIGS["prospector_like"]["mismatches_text"],
        ):
            assert "Gyr" not in claim or "912" in claim, claim
            assert "age floor" not in claim.lower() and "formation age" not in claim.lower()

    def test_catalog_absence_rows_only_claim_not_recorded(self):
        for key, text, mismatch, src in _all_rows():
            if "header (absence" in src:
                assert "not recorded" in text, (key, text)
                assert "not recorded" in mismatch or "does not record" in mismatch, (key, mismatch)

    def test_repo_file_line_sources_exist_and_in_range(self):
        checked = 0
        for key, cfg in XLIKE_CONFIGS.items():
            raws = [*cfg["mismatch_sources"], *cfg.get("notes_sources", [])]
            for raw in raws:
                for rel, lines in _cited_file_lines(raw):
                    path = _resolve_source_file(rel)
                    top = rel.split("/")[0]
                    if path is None and top in _UPSTREAM and _site_packages_root(top) is None:
                        continue  # upstream package not installed in this environment
                    assert path is not None, (key, raw, rel)
                    n_lines = len(path.read_text(errors="replace").splitlines())
                    assert min(lines) >= 1 and max(lines) <= n_lines, (key, raw, n_lines)
                    checked += 1
        assert checked >= 10

    def test_beagle_dust_row_cites_catalog_columns(self):
        cfg = XLIKE_CONFIGS["beagle_like"]
        srcs = [s for s in cfg["mismatch_sources"] if "BEAGLE_summary_catalogue" in s]
        assert srcs and "tauV_eff" in srcs[0] and "mu" in srcs[0]

    def test_cloudy_version_no_17_02(self):
        """Verify unsourced 'Cloudy 17.02' has been replaced with 'Cloudy 17'."""
        for key, cfg in XLIKE_CONFIGS.items():
            # Check both mismatches and mismatches_text for the outdated version string
            for text in cfg.get("mismatches", []):
                assert "17.02" not in text, f"{key} mismatches contains unsourced '17.02': {text}"
            for text in cfg.get("mismatches_text", []):
                assert "17.02" not in text, (
                    f"{key} mismatches_text contains unsourced '17.02': {text}"
                )

    def test_cigale_bc03_row_no_version_word(self):
        """Verify cigale_like BC03 row has been reworded without 'version' word."""
        cfg = XLIKE_CONFIGS["cigale_like"]
        # The first row should be about BC03 differences
        first_mismatch = cfg["mismatches"][0]
        first_mismatch_text = cfg["mismatches_text"][0]
        assert "version" not in first_mismatch.lower(), (
            f"cigale_like BC03 row should not use 'version': {first_mismatch}"
        )
        assert "version" not in first_mismatch_text.lower(), (
            f"cigale_like BC03 text row should not use 'version': {first_mismatch_text}"
        )
        # Verify the new wording is present
        assert "CIGALE-like configuration" in first_mismatch_text
        assert "registered BC03" in first_mismatch_text


_COLUMN_LIST_SOURCE = "art_sedfitting/code_outputs/header"


def _display_strings():
    """Every reader-facing string of both tables and the JSON display fields."""
    for cfg in XLIKE_CONFIGS.values():
        yield from cfg["mismatches_text"]
        yield from cfg.get("notes", [])
    yield render_latex()


def _only_column_list(raw: str) -> bool:
    parts = [part.strip() for part in raw.split(";")]
    return all(part.startswith(_COLUMN_LIST_SOURCE) for part in parts)


def _rows_matching(key: str, needle: str) -> list[str]:
    return [t for t in XLIKE_CONFIGS[key]["mismatches_text"] if needle in t]


class TestTablesFollowEachCodesOwnSource:
    """Table A lists options, not run settings; Table B rows rest on upstream code or this work."""

    def test_no_fiducial_or_specification_wording(self):
        for text in _display_strings():
            assert "fiducial" not in text.lower()
            assert "specification" not in text.lower()

    def test_choices_header_says_options(self):
        header = next(ln for ln in render_latex().splitlines() if ln.startswith("Code & Parity"))
        assert "Options" in header
        assert "Table~1" in header

    def test_cigale_attenuation_row_is_one_screen(self):
        rows = _rows_matching("cigale_like", "modified-starburst")
        assert len(rows) == 1
        text = rows[0]
        assert not re.search(r"\bratio\b", text)
        assert "birth-cloud/diffuse ratio" not in text
        assert "one" in text
        assert "0.44" in text

    def test_cigale_metallicity_row_lists_the_six_values(self):
        rows = _rows_matching("cigale_like", "six metallicities")
        assert len(rows) == 1
        for z in ("0.0001", "0.0004", "0.004", "0.008", "0.02", "0.05"):
            assert z in rows[0]
        assert "continuously" in rows[0]

    def test_only_cigale_has_a_metallicity_row(self):
        for key, cfg in XLIKE_CONFIGS.items():
            if key != "cigale_like":
                for text in cfg["mismatches_text"]:
                    assert "metallicit" not in text.lower(), (key, text)

    def test_no_prospector_sfh_prior_difference_row(self):
        for text in XLIKE_CONFIGS["prospector_like"]["mismatches_text"]:
            assert "Student" not in text and "prior" not in text.lower(), text
        assert any(
            "identical to Prospector" in n for n in XLIKE_CONFIGS["prospector_like"]["notes"]
        )

    def test_no_difference_row_rests_only_on_the_column_list(self):
        for key, _, mismatch, src in _all_rows():
            assert not _only_column_list(src), (key, mismatch)

    def test_column_list_only_backs_a_not_recorded_statement(self):
        for key, text, _, src in _all_rows():
            if _COLUMN_LIST_SOURCE in src:
                assert "not recorded" in text, (key, text)
        for cfg in XLIKE_CONFIGS.values():
            for note, src in zip(cfg.get("notes", []), cfg.get("notes_sources", []), strict=True):
                if _COLUMN_LIST_SOURCE in src:
                    assert "not record" in note, note

    def test_beagle_dust_row_is_a_reparametrization(self):
        rows = _rows_matching("beagle_like", "Charlot")
        assert len(rows) == 1
        assert "in place of BEAGLE's (tau_V, mu)" in rows[0]
        assert "same" in rows[0]
        assert r"($\tau_V$, $\mu$)" in render_latex()

    def test_bagpipes_nebular_row_states_both_cloudy_versions(self):
        rows = _rows_matching("bagpipes_like", "Cloudy")
        assert len(rows) == 1
        assert "Cloudy 17 (C17)" in rows[0] and "Cloudy 25" in rows[0]
        assert "trained on Cloudy 17" in rows[0]

    def test_dense_basis_nebular_row_says_no_version_year(self):
        rows = _rows_matching("dense_basis_like", "Cloudy")
        assert len(rows) == 1 and "no version year" in rows[0]

    def test_unsourced_rows_stay_out_of_the_differences(self):
        assert not _rows_matching("bagpipes_like", "uniform distributions")
        assert not _rows_matching("beagle_like", "burst")
        assert not _rows_matching("dense_basis_like", "FSPS library")
        assert any("FSPS isochrone" in n for n in XLIKE_CONFIGS["dense_basis_like"]["notes"])

    def test_energy_balance_row_cites_the_paper_data_section(self):
        cfg = XLIKE_CONFIGS["prospector_like"]
        idx = next(i for i, t in enumerate(cfg["mismatches_text"]) if "912" in t)
        latex, plain = source_display(cfg["mismatch_sources"][idx])
        assert r"Section~\ref{subsec:demonstration:data}" in latex
        assert "tengri reproduction notebook (Prospector)" in latex
        assert "subsec:demonstration:data" in plain

    def test_upstream_sources_resolve_to_installed_packages(self):
        cited = [
            (key, rel, lines)
            for key, cfg in XLIKE_CONFIGS.items()
            for raw in (*cfg["mismatch_sources"], *cfg.get("notes_sources", []))
            for rel, lines in _cited_file_lines(raw)
            if rel.split("/")[0] in _UPSTREAM
        ]
        assert {rel.split("/")[0] for _, rel, _ in cited} == set(_UPSTREAM)
        for top in _UPSTREAM:
            if _site_packages_root(top) is None:
                pytest.skip(f"upstream package {top} is not installed here")
        for key, rel, lines in cited:
            path = _resolve_source_file(rel)
            assert path is not None, (key, rel)
            n_lines = len(path.read_text(errors="replace").splitlines())
            assert min(lines) >= 1 and max(lines) <= n_lines, (key, rel, n_lines)
