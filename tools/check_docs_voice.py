#!/usr/bin/env python3
"""Docs prose voice guard — enforcement of scientific writing standards.

This guard enforces seven writing conventions for user-facing documentation:

1. **Restatement openings** (gallery only): Body first sentence must not parrot
   the title with "Demonstrates", "Shows how", etc.
2. **Fact-free bulk**: Prose >= 120 words with no unit, number, citation, or caveat
   (must, never, silently, fails, etc.) is likely padding.
3. **Dev-detail leakage**: Issue references (#123), private vars (_REGISTRY),
   and references to excluded doc trees.
4. **Unicode normalization**: Non-normalized codepoints (MICRO SIGN, ligatures,
   superscript digits) signal encoding drift.
5. **Notation canon**: Standardized forms (μm, Å, M☉, τ_V, χ², S/N, en-dashes,
   reference-code casing) vs non-standard variants.
6. **Hardcoded constants**: Literal values exported by physics_constants.py must
   not reappear in published docs (rules #1277, #1749, #1752).
7. **Prose vs printed output**: Evaluative prose (\"recovers\", \"agrees\", etc.)
   adjacent to code cell outputs characterizes what the reader can see —
   violation of the verdict rule.

Reported metrics (no exit 1):
- Coefficient of variation of section/docstring word-length per corpus.
- Distribution of gallery header lengths (count >= 120 words).
- Hand-holding percentage per notebook.

Corpus: examples/*/plot_*.py (sphinx-gallery headers + %% blocks), examples/README.rst,
notebooks/[0-9]*.py (jupytext %% blocks), docs/*/index.md, docs/units.md, docs/spine/*.ipynb
(markdown cells only), docs/reproduction/*.ipynb (markdown cells).

Excluded: docs/_build, docs/auto_examples, docs/dev, docs/adr (generated and design notes).

Usage
-----
    python tools/check_docs_voice.py            # scan + report metrics (exit 0 clean)
    python tools/check_docs_voice.py --strict   # fail on any violation
    python tools/check_docs_voice.py --report   # metrics only, no gated checks
    python tools/check_docs_voice.py --fix      # rewrite notation_canon + unicode_norm in place
    python tools/check_docs_voice.py --fix --dry-run  # preview fixes without writing

CI: not wired — warn-only until the backlog it reports is cleared (#2050).

"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Constants for corpus discovery
GALLERY_PATTERN = r"examples/*/plot_*.py"
NOTEBOOK_PATTERN = r"notebooks/[0-9]*.py"
SPINE_IPYNB = r"docs/spine/*.ipynb"
REPRO_IPYNB = r"docs/reproduction/*.ipynb"
DOC_MARKDOWN = [
    Path("docs/index.md"),
    Path("docs/overview.md"),
    Path("docs/units.md"),
    Path("docs/reproduction/index.md"),
]

EXCLUDE_DIRS = {"docs/_build", "docs/build", "docs/auto_examples", "docs/dev", "docs/adr"}

# Notation canon: canonical form -> list of non-canonical variants to flag
NOTATION_CANON = {
    "μm": ["um", "micron", "microns"],  # U+03BC GREEK SMALL LETTER MU
    "Å": ["Angstrom", "Angstroms", "AA"],
    "M☉": ["Msun", "M_sun", "Msol"],
    "Z☉": ["Zsun", "Z_sun"],
    "χ²": ["chi2", "chi^2"],
    "S/N": ["SNR", "snr="],
    "τ_diff": ["tau_diff"],
    "τ_bc": ["tau_bc"],
    "τ_V": ["tau_V"],
    "Hα": ["H_alpha", "Halpha", "H-alpha", "H$\\alpha$"],
    "R̂": ["r_hat", "rhat", "Rhat"],
    "log₁₀": ["log_10"],
}

# Reference code canonical casing
REF_CODE_CANON = {"BAGPIPES", "Prospector", "CIGALE", "AGNfitter", "Synthesizer", "FSPS", "DSPS"}

# Caveat keywords: indicate substantive content, not padding
CAVEAT_KEYWORDS = {"must", "never", "silently", "fails", "raises", "not", "unless", "only", "wrong", "invalid", "requires"}

# Unit tokens to detect fact-ful content
UNIT_TOKENS = {"Å", "μm", "erg", "Msun", "M☉", "mag", "Gyr", "Myr", "yr", "dex", "K", "Jy", "keV", "GHz"}

# Restatement opening patterns (gallery headers only)
RESTATEMENT_PATTERNS = {
    "demonstrates", "generates", "we build", "we construct", "computes",
    "shows how", "this example", "this script", "illustrates"
}

# STRONG evaluative tokens: characterize a quantity the reader can see printed.
# Drop weak tokens ("agree", "matches", "close to") that describe setup, not verdict.
EVALUATIVE_TOKENS_STRONG = {
    "recovers", "recover",
    "reproduce well", "reproduces well",
    "well-mixed", "well constrained",
    "good agreement",
    "track the", "tracks the",
    "excellent"
}

# Development-detail patterns to flag
DEV_PATTERNS = {
    "issue_ref": re.compile(r"#\d{3,4}"),
    "claude_md": re.compile(r"\bCLAUDE\.md\b"),
    "agents_md": re.compile(r"\bAGENTS\.md\b"),
    # Not preceded by an identifier char or a closing bracket: physics prose
    # writes subscripts like E(B-V)_BBB and sigma_NMAD, which are notation,
    # not private Python names.
    "private_var": re.compile(r"(?<![A-Za-z0-9)\]])\b_[A-Z][A-Z0-9_]{2,}\b"),
    "dev_path": re.compile(r"\bdocs/(dev|adr)/"),
}

# Hardcoded allowlist (legitimate private vars)
PRIVATE_VAR_ALLOWLIST = {"SFH_REGISTRY", "_REGISTRY"}


@dataclass
class Violation:
    """A single prose violation."""
    path: str
    line: int
    col: int
    check_name: str
    text: str
    message: str


def _extract_constants_from_physics() -> dict[str, float]:
    """Import physics_constants module to get all exported float/int constants.

    Reads module-level UPPER_SNAKE attributes, including computed constants like
    LOG10_ZSUN = math.log10(Z_SUN). Falls back gracefully if tengri is not installed.
    """
    constants = {}

    # Try to import the actual module (handles computed constants, cannot drift from source)
    try:
        import importlib
        physics_module = importlib.import_module("tengri.utils.physics_constants")
        for name in dir(physics_module):
            # Only UPPER_SNAKE public attributes (not methods, not __dunder__)
            if name.isupper() and not name.startswith("_"):
                val = getattr(physics_module, name)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    constants[name] = float(val)
        if constants:
            return constants
    except ImportError:
        # Fallback: warn but don't crash
        print(
            "WARNING: tengri.utils.physics_constants not importable; "
            "hardcoded_const check will be incomplete",
            file=sys.stderr
        )
    except Exception as e:
        print(f"WARNING: Failed to import physics constants: {e}", file=sys.stderr)

    # Fallback: parse the file directly if import fails
    physics_file = REPO / "src" / "tengri" / "utils" / "physics_constants.py"
    if physics_file.exists():
        try:
            content = physics_file.read_text(encoding="utf-8")
            for line in content.split("\n"):
                # Match: NAME: float = <literal>
                m = re.match(r"^(\w+):\s*float\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
                if m:
                    name, val = m.groups()
                    constants[name] = float(val)
        except Exception:
            pass

    return constants


def _mask_code_spans(text: str) -> str:
    """Blank out code spans so identifiers are not flagged as prose.

    Masks:
    - Markdown/MyST: fenced blocks (``` and ~~~), inline (`` `x` ``)
    - RST: ``code``, .. code-block::, indented literal blocks
    - Python docstrings: same RST rules
    """
    # Blank markdown/MyST fenced blocks
    text = re.sub(r"```[\s\S]*?```", lambda m: "\n" * m.group(0).count("\n"), text)
    text = re.sub(r"~~~[\s\S]*?~~~", lambda m: "\n" * m.group(0).count("\n"), text)

    # Blank RST code-block directives and following indented content
    text = re.sub(r"\.\.\s+code-block::[^\n]*\n((?:\s{3,}[^\n]*\n)*)", lambda m: "\n" * m.group(0).count("\n"), text)

    # Blank RST literal blocks (text followed by ::)
    text = re.sub(r"::\n\n((?:[ ]{3,}[^\n]*\n)+)", lambda m: "\n" * m.group(0).count("\n"), text)

    # Blank inline code: ``x`` and `x`
    text = re.sub(r"``[^`]+``", lambda m: "x" * len(m.group(0)), text)
    text = re.sub(r"`[^`]+`", lambda m: "x" * len(m.group(0)), text)

    return text


def _extract_markdown_cells_from_ipynb(path: Path) -> list[tuple[int, str, bool]]:
    """Extract markdown cell contents from .ipynb file.

    Returns list of (cell_index, markdown_text, adjacent_code_cell_has_output) tuples.

    A code cell "has output" if it has execution_count (ran) and has output items.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cells = []
        all_cells = data.get("cells", [])

        for i, cell in enumerate(all_cells):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                if isinstance(source, list):
                    text = "".join(source)
                else:
                    text = source

                # Check if adjacent code cell has output
                adjacent_has_output = False
                # Check next cell (after markdown)
                if i + 1 < len(all_cells):
                    next_cell = all_cells[i + 1]
                    if next_cell.get("cell_type") == "code":
                        outputs = next_cell.get("outputs", [])
                        has_execution = next_cell.get("execution_count") is not None
                        if has_execution and outputs:
                            # Check if any output has actual content (not just empty)
                            for out in outputs:
                                if out.get("output_type") in ("execute_result", "display_data"):
                                    adjacent_has_output = True
                                    break

                # Also check previous cell (before markdown)
                if i - 1 >= 0:
                    prev_cell = all_cells[i - 1]
                    if prev_cell.get("cell_type") == "code":
                        outputs = prev_cell.get("outputs", [])
                        has_execution = prev_cell.get("execution_count") is not None
                        if has_execution and outputs:
                            for out in outputs:
                                if out.get("output_type") in ("execute_result", "display_data"):
                                    adjacent_has_output = True
                                    break

                cells.append((i, text, adjacent_has_output))
        return cells
    except Exception:
        return []


def _extract_gallery_header(path: Path) -> str | None:
    """Extract sphinx-gallery header docstring from plot_*.py file."""
    try:
        text = path.read_text(encoding="utf-8")
        # Find first triple-quoted block
        m = re.match(r'"""(.*?)"""', text, re.DOTALL)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None


def _extract_percent_blocks(path: Path) -> list[tuple[int, str]]:
    """Extract # %% [markdown] blocks from jupytext .py file.

    Returns list of (line_number, markdown_text) tuples.
    A markdown block starts at `# %% [markdown]` and continues through
    #-prefixed lines until the next `# %%` delimiter or end of file.
    """
    blocks = []
    try:
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            # Match `# %% [markdown]` exactly (with optional spacing)
            if re.match(r"^#\s*%%\s*\[markdown\]\s*$", lines[i]):
                block_start_line = i
                block_lines = []
                i += 1
                # Collect all #-prefixed lines until next %% delimiter
                while i < len(lines):
                    # Check if we've hit the next cell delimiter
                    if re.match(r"^#\s*%%", lines[i]):
                        break
                    # Only collect #-prefixed comment lines
                    if lines[i].startswith("#"):
                        line = lines[i]
                        # Strip leading "# " or "#"
                        if line.startswith("# "):
                            block_lines.append(line[2:])
                        elif line == "#":
                            block_lines.append("")
                        else:
                            block_lines.append(line[1:])
                        i += 1
                    else:
                        # Non-comment line inside markdown block (shouldn't happen in well-formed files)
                        break
                if block_lines:
                    blocks.append((block_start_line, "\n".join(block_lines)))
            else:
                i += 1
    except Exception:
        pass
    return blocks


def _check_restatement_openings(text: str) -> list[str]:
    """Check for restatement openings in gallery header docstrings."""
    violations = []
    if not text:
        return violations

    sentences = re.split(r"[.!?]+", text)
    if len(sentences) < 2:
        return violations

    # Check the second sentence (first sentence after title)
    body_start = sentences[1].strip().lower()
    for pattern in RESTATEMENT_PATTERNS:
        if body_start.startswith(pattern):
            violations.append(f"Body begins with '{pattern}': {text[:100]}")

    return violations


def _check_fact_free_bulk(text: str) -> list[str]:
    """Flag prose >= 120 words with no unit, number, citation, or caveat."""
    violations = []

    # Count words
    words = len(text.split())
    if words < 120:
        return violations

    masked = _mask_code_spans(text)

    # Check for facts: digits, units, citations, caveats
    has_digit = bool(re.search(r"\d", masked))
    has_unit = any(unit in masked for unit in UNIT_TOKENS)
    has_citation = bool(re.search(r"\w+\+\d{4}|et al", masked))
    has_caveat = any(f" {kw} " in masked.lower() or f" {kw}." in masked.lower() for kw in CAVEAT_KEYWORDS)

    if not (has_digit or has_unit or has_citation or has_caveat):
        violations.append(f"Fact-free padding: {words} words, no units/numbers/citations/caveats")

    return violations


def _check_dev_detail_leakage(text: str) -> list[str]:
    """Flag development details leaking into published prose."""
    violations = []
    masked = _mask_code_spans(text)

    if DEV_PATTERNS["issue_ref"].search(masked):
        violations.append("Issue reference #NNN found")
    if DEV_PATTERNS["claude_md"].search(masked):
        violations.append("CLAUDE.md reference found")
    if DEV_PATTERNS["agents_md"].search(masked):
        violations.append("AGENTS.md reference found")
    # A full github.com blob URL is the sanctioned way to point at an unpublished
    # tree -- readme.md does exactly this. Only bare site-relative paths are dead
    # links, so strip absolute URLs before looking for them.
    without_urls = re.sub(r"https?://\S+", "", masked)
    if DEV_PATTERNS["dev_path"].search(without_urls):
        violations.append("Reference to docs/dev or docs/adr found")

    for m in DEV_PATTERNS["private_var"].finditer(masked):
        var = m.group(0)
        if var not in PRIVATE_VAR_ALLOWLIST:
            violations.append(f"Private var '{var}' found")

    return violations


def _check_unicode_normalization(text: str) -> list[str]:
    """Flag specific confusable codepoints that NFKC targets.

    Allowlist intentional typography: superscripts/subscripts (²³₀₁...₉),
    ellipsis (…), and anything in the notation canon table.

    Target confusables: MICRO SIGN (U+00B5), ligatures (ﬁ ﬂ), non-breaking spaces,
    Cyrillic/Greek homoglyphs of Latin letters, full-width forms.
    """
    violations = []

    # Codepoints that are intentionally formatted and should not be normalized
    INTENTIONAL = {
        "²", "³", "¹",  # superscripts
        "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉",  # subscripts
        "…",  # ellipsis
        "χ", "²",  # from notation canon (χ²)
        "₁", "₀",  # from notation canon (log₁₀)
        "Å", "☉", "×", "–", "μ", "τ", "Hα",  # from canon: all NFKC-stable
        "R̂",  # combining circumflex: NFKC-stable
    }

    # The ONE problematic normalization: U+00B5 MICRO SIGN → U+03BC GREEK SMALL LETTER MU
    CONFUSABLES_TO_FLAG = {
        "µ": ("MICRO SIGN", "μ"),  # U+00B5 → U+03BC (the only one we care about)
    }

    for i, char in enumerate(text):
        if char in CONFUSABLES_TO_FLAG:
            confusable_name, nfkc_form = CONFUSABLES_TO_FLAG[char]
            violations.append(
                f"Confusable codepoint {confusable_name} (U+{ord(char):04X}); "
                f"normalizes to U+{ord(nfkc_form):04X}"
            )

    # Also flag ligatures and other genuine confusables
    ligatures = {"ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ"}
    for char in ligatures:
        if char in text:
            try:
                char_name = unicodedata.name(char)
            except ValueError:
                char_name = f"U+{ord(char):04X}"
            violations.append(f"Ligature {char_name} should be expanded")

    return violations


def _check_notation_canon(text: str) -> list[str]:
    """Flag non-canonical notation forms."""
    violations = []
    masked = _mask_code_spans(text)

    for canonical, variants in NOTATION_CANON.items():
        for variant in variants:
            if variant in masked:
                # Ensure it's standalone (word boundaries for text, not for symbols)
                if re.search(rf"\b{re.escape(variant)}\b", masked):
                    violations.append(f"Non-canonical '{variant}' (should be '{canonical}')")

    # Check reference code casing
    for word in re.findall(r"\b[A-Za-z]+\b", masked):
        if word in REF_CODE_CANON or any(word.lower() == ref.lower() for ref in REF_CODE_CANON):
            if word not in REF_CODE_CANON:
                canonical = next(ref for ref in REF_CODE_CANON if ref.lower() == word.lower())
                violations.append(f"Non-canonical casing '{word}' (should be '{canonical}')")

    # Check en-dashes in numeric ranges: flag hyphen between numbers. ISO dates
    # (2026-04-08, 2026-05) are not ranges and must keep their hyphens, so drop
    # them before looking. Same for version strings like 0.4.6-rc1.
    no_dates = re.sub(r"\b\d{4}-\d{2}(-\d{2})?\b", "", masked)
    if re.search(r"\d+-\d+", no_dates):
        violations.append("Numeric range with hyphen (should be en-dash –)")

    return violations


def _check_hardcoded_constants_in_code(text: str, constants: dict[str, float]) -> list[str]:
    """Flag CODE ASSIGNMENTS of constant literals matching physics_constants.py exports.

    Only checks assignment patterns: NAME = <literal>. Prose mentions of values do NOT count.
    Requires: exact match OR rounded form within matching tolerance.
    Example violations: LOG10_ZSUN = -1.848 in a code cell (matches -1.8477...)

    Suppression: A trailing `# docs-const: intentional — <reason>` pragma suppresses the finding.
    The pragma must be explicit and greppable; it is not auto-suppressed by any comment.
    """
    violations = []

    # Extract assignment patterns from text: VAR_NAME = <number>
    # This regex requires the assignment to be actual code, not prose
    assignment_pattern = re.compile(
        r"^(\w+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(.*?)$",
        re.MULTILINE
    )

    for match in assignment_pattern.finditer(text):
        var_name, literal_str, rest_of_line = match.groups()
        try:
            literal_val = float(literal_str)
        except ValueError:
            continue

        # Check for suppression pragma on this line
        if "# docs-const: intentional —" in rest_of_line:
            continue

        # STRICT: var name must contain the constant name (case-insensitive)
        # AND the value must be within 0.5% of the constant value
        for const_name, const_val in constants.items():
            # Must appear in variable name
            if const_name.lower() not in var_name.lower():
                continue

            # Skip zero values (too many false positives)
            if const_val == 0 or literal_val == 0:
                continue

            # Tolerance: 0.5% for near matches
            tol = abs(const_val) * 0.005
            if abs(literal_val - const_val) <= tol:
                violations.append(
                    f"Hardcoded assignment {var_name} = {literal_str} matches constant "
                    f"{const_name} (import from physics_constants.py)"
                )
                break

    return violations


def _fix_notation_canon(text: str) -> str:
    """Fix non-canonical notation forms to canonical ones.

    Rewrites in place, respecting code spans. Handles:
    - Notation variants (um -> μm, SNR -> S/N, etc.)
    - Reference code casing (bagpipes -> Bagpipes, etc.)
    - Hyphen ranges between numbers (16-84% -> 16–84%)

    A ``**Verification Status:**`` line is exempt. Its component name is a key
    matched verbatim against the table in docs/dev/verification-protocol.md, not
    prose, so applying reference-code casing to it breaks the match: the protocol
    row reads "Dust IR emission vs bagpipes" and canonicalizing that to BAGPIPES
    made check_repro_status fail with "unknown component".
    """
    lines = text.split("\n")
    if any("**Verification Status:**" in ln for ln in lines):
        return "\n".join(
            ln if "**Verification Status:**" in ln else _fix_notation_canon(ln)
            for ln in lines
        )

    # A toctree body is a list of document names, not prose. Applying
    # reference-code casing to it renames the target: `bagpipes` became
    # `BAGPIPES` and sphinx failed with "toctree contains reference to
    # nonexisting document". Same shape as the Verification Status case above --
    # prose rules must not reach anything matched as an identifier.
    if "{toctree}" in text or ".. toctree::" in text:
        out, in_toctree = [], False
        for ln in lines:
            if "{toctree}" in ln or ".. toctree::" in ln:
                in_toctree = True
                out.append(ln)
                continue
            if in_toctree:
                if ln.strip().startswith("```") or (ln.strip() and not ln.startswith((" ", "\t", ":"))
                                                    and ".. toctree::" in text and not ln.strip()):
                    in_toctree = False
                elif ln.strip() == "" and out and out[-1].strip() == "":
                    in_toctree = False
                out.append(ln)
                continue
            out.append(_fix_notation_canon(ln))
        return "\n".join(out)

    masked = _mask_code_spans(text)

    def is_in_code(pos: int, length: int) -> bool:
        """Check if any part of [pos, pos+length) overlaps with code."""
        for i in range(pos, min(pos + length, len(masked))):
            if i < len(masked) and masked[i] == 'x':
                return True
        return False

    # Collect ALL replacements based on ORIGINAL text
    replacements = []

    # 1. Fix notation variants (using NOTATION_CANON)
    for canonical, variants in NOTATION_CANON.items():
        for variant in variants:
            pattern = re.compile(rf"\b{re.escape(variant)}\b")
            for match in pattern.finditer(text):
                if not is_in_code(match.start(), len(variant)):
                    replacements.append((match.start(), match.end(), canonical))

    # 2. Fix reference code casing
    pattern = re.compile(r"\b[A-Za-z]+\b")
    for match in pattern.finditer(text):
        word = match.group(0)
        if any(word.lower() == ref.lower() for ref in REF_CODE_CANON):
            if word not in REF_CODE_CANON:
                canonical = next(ref for ref in REF_CODE_CANON if ref.lower() == word.lower())
                if not is_in_code(match.start(), len(word)):
                    replacements.append((match.start(), match.end(), canonical))

    # 3. Fix numeric ranges (hyphen to en-dash)
    # First, mark positions of ISO dates and version strings to skip
    skip_ranges = []
    for m in re.finditer(r"\b\d{4}-\d{2}(-\d{2})?\b", text):
        skip_ranges.append((m.start(), m.end()))
    for m in re.finditer(r"(\d+\.)+\d+-[a-zA-Z0-9]+", text):
        skip_ranges.append((m.start(), m.end()))

    def overlaps_skip(pos: int) -> bool:
        """Check if position overlaps with any skip range."""
        return any(s <= pos < e for s, e in skip_ranges)

    pattern = re.compile(r"(\d+)-(\d+)")
    for match in pattern.finditer(text):
        # Find the hyphen position within the match
        hyphen_idx = match.group(0).find('-')
        hyphen_pos = match.start() + hyphen_idx
        # Only replace if not in a skip range and not in code
        if not overlaps_skip(hyphen_pos) and not is_in_code(hyphen_pos, 1):
            replacements.append((hyphen_pos, hyphen_pos + 1, "–"))

    # Apply all replacements in reverse order to maintain positions
    result = text
    for start, end, replacement in sorted(replacements, reverse=True):
        result = result[:start] + replacement + result[end:]

    return result


def _fix_unicode_normalization(text: str) -> str:
    """Fix confusable unicode codepoints to canonical forms.

    Replaces:
    - U+00B5 MICRO SIGN -> U+03BC GREEK SMALL LETTER MU (μ)
    - Ligatures (ﬁ, ﬂ, etc.) to their ASCII expansions
    """
    # Replace MICRO SIGN with GREEK SMALL LETTER MU
    text = text.replace("µ", "μ")  # U+00B5 -> U+03BC

    # Replace ligatures
    ligature_map = {
        "ﬁ": "fi",   # U+FB01 LATIN SMALL LIGATURE FI
        "ﬂ": "fl",   # U+FB02 LATIN SMALL LIGATURE FL
        "ﬀ": "ff",   # U+FB00 LATIN SMALL LIGATURE FF
        "ﬃ": "ffi",  # U+FB03 LATIN SMALL LIGATURE FFI
        "ﬄ": "ffl",  # U+FB04 LATIN SMALL LIGATURE FFL
    }

    for ligature, replacement in ligature_map.items():
        text = text.replace(ligature, replacement)

    return text


def _check_prose_vs_output(text: str, adjacent_cell_has_output: bool) -> list[str]:
    """Flag strong evaluative tokens in prose adjacent to code cell outputs.

    Violation: prose contains a verdict token (recovers, matches, well-mixed, etc.)
    in a markdown cell that immediately precedes/follows a code cell with printed output.

    Only flags STRONG tokens that characterize a number the reader can see.
    Drops weak tokens like "agree"/"matches"/"close to" (used for setup description).

    Suppression: a ``<!-- docs-voice: criterion -->`` marker anywhere in the cell
    suppresses the finding. The distinction this check cannot make is semantic:
    "these chains are well-mixed" grades the run in front of the reader, while
    "well-mixed chains overlap and look like white noise" teaches them how to read
    the plot and is true of every run. The second is what the docs are for. Without
    an escape hatch the only prose that reliably passes is prose that says nothing,
    so the marker is deliberate, greppable, and belongs only on durable criteria.
    """
    violations = []

    if not adjacent_cell_has_output:
        return violations

    if "<!-- docs-voice: criterion -->" in text:
        return violations

    # Interrogative lines pose the question the printed output answers -- a
    # heading like "Did the parallel fit recover the truth?" asserts nothing.
    text_lower = "\n".join(
        line for line in text.lower().splitlines() if not line.rstrip().endswith("?")
    )
    for token in EVALUATIVE_TOKENS_STRONG:
        # Word-boundary match: substring matching flags "recovery table" on the
        # token "recover", which is a noun phrase naming a table, not a verdict.
        if re.search(rf"\b{re.escape(token)}\b", text_lower):
            violations.append(
                f"Evaluative token '{token}' in prose adjacent to code output "
                f"(verdict must not characterize what reader can see printed)"
            )

    return violations


def scan_file(path: Path, constants: dict[str, float]) -> list[Violation]:
    """Scan one file for all prose violations."""
    violations = []

    if path.suffix == ".rst":
        # RST files: extract prose sections, skip code blocks
        try:
            text = path.read_text(encoding="utf-8")
            masked = _mask_code_spans(text)

            # Split into paragraphs
            for para_idx, para in enumerate(masked.split("\n\n")):
                if not para.strip() or len(para.split()) < 3:
                    continue

                line_num = masked[:masked.find(para)].count("\n") + 1
                col = 1

                for check_name, check_fn in [
                    ("fact_free", _check_fact_free_bulk),
                    ("dev_detail", _check_dev_detail_leakage),
                    ("unicode_norm", _check_unicode_normalization),
                    ("notation_canon", _check_notation_canon),
                ]:
                    for msg in check_fn(para):
                        violations.append(Violation(
                            str(path.relative_to(REPO)), line_num, col, check_name, para[:50], msg
                        ))
        except Exception:
            pass

    elif path.suffix == ".md":
        # Markdown files
        try:
            text = path.read_text(encoding="utf-8")
            masked = _mask_code_spans(text)

            for para_idx, para in enumerate(masked.split("\n\n")):
                if not para.strip():
                    continue

                line_num = masked[:masked.find(para)].count("\n") + 1
                col = 1

                for check_name, check_fn in [
                    ("fact_free", _check_fact_free_bulk),
                    ("dev_detail", _check_dev_detail_leakage),
                    ("unicode_norm", _check_unicode_normalization),
                    ("notation_canon", _check_notation_canon),
                ]:
                    for msg in check_fn(para):
                        violations.append(Violation(
                            str(path.relative_to(REPO)), line_num, col, check_name, para[:50], msg
                        ))
        except Exception:
            pass

    elif "plot_" in path.name and path.suffix == ".py":
        # Gallery example: check header + %% blocks
        header = _extract_gallery_header(path)
        if header:
            line_num = 1
            for msg in _check_restatement_openings(header):
                violations.append(Violation(
                    str(path.relative_to(REPO)), line_num, 1, "restatement_opening", header[:50], msg
                ))

            masked_header = _mask_code_spans(header)
            for check_name, check_fn in [
                ("fact_free", _check_fact_free_bulk),
                ("dev_detail", _check_dev_detail_leakage),
                ("unicode_norm", _check_unicode_normalization),
                ("notation_canon", _check_notation_canon),
            ]:
                for msg in check_fn(masked_header):
                    violations.append(Violation(
                        str(path.relative_to(REPO)), line_num, 1, check_name, header[:50], msg
                    ))

        # Check %% markdown blocks
        blocks = _extract_percent_blocks(path)
        for block_line_num, block_content in blocks:
            masked_block = _mask_code_spans(block_content)
            for check_name, check_fn in [
                ("fact_free", _check_fact_free_bulk),
                ("dev_detail", _check_dev_detail_leakage),
                ("unicode_norm", _check_unicode_normalization),
                ("notation_canon", _check_notation_canon),
                ("hardcoded_const", lambda t: _check_hardcoded_constants(t, constants)),
            ]:
                for msg in check_fn(masked_block):
                    violations.append(Violation(
                        str(path.relative_to(REPO)), block_line_num, 1, check_name, block_content[:50], msg
                    ))

    elif path.suffix == ".ipynb":
        # Jupyter notebook: scan markdown cells and code cells (for hardcoded constants)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            all_cells = data.get("cells", [])

            markdown_cells = _extract_markdown_cells_from_ipynb(path)
            for cell_idx, cell_text, adjacent_has_output in markdown_cells:
                masked_cell = _mask_code_spans(cell_text)

                for check_name, check_fn in [
                    ("fact_free", _check_fact_free_bulk),
                    ("dev_detail", _check_dev_detail_leakage),
                    ("unicode_norm", _check_unicode_normalization),
                    ("notation_canon", _check_notation_canon),
                    ("prose_vs_output", lambda t: _check_prose_vs_output(t, adjacent_has_output)),
                ]:
                    for msg in check_fn(masked_cell):
                        violations.append(Violation(
                            str(path.relative_to(REPO)), cell_idx, 1, check_name, cell_text[:50], msg
                        ))

            # Check code cells for hardcoded constants (check 6)
            for cell_idx, cell in enumerate(all_cells):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    if isinstance(source, list):
                        code_text = "".join(source)
                    else:
                        code_text = source

                    for msg in _check_hardcoded_constants_in_code(code_text, constants):
                        violations.append(Violation(
                            str(path.relative_to(REPO)), cell_idx, 1, "hardcoded_const", code_text[:50], msg
                        ))
        except Exception:
            pass

    return violations


def iter_corpus() -> list[Path]:
    """Iterate over all files in the user-facing corpus."""
    files = []

    # Gallery examples
    files.extend(REPO.glob("examples/*/plot_*.py"))

    # Example READMEs
    if (REPO / "examples" / "README.rst").exists():
        files.append(REPO / "examples" / "README.rst")
    files.extend(REPO.glob("examples/*/README.rst"))

    # Notebooks
    files.extend(REPO.glob("notebooks/[0-9]*.py"))

    # Docs markdown
    for md_path in DOC_MARKDOWN:
        if (REPO / md_path).exists():
            files.append(REPO / md_path)

    # Spine and reproduction notebooks
    files.extend(REPO.glob("docs/spine/*.ipynb"))
    files.extend(REPO.glob("docs/reproduction/*.ipynb"))

    # Filter out excluded directories
    return [f for f in files if not any(excl in f.as_posix() for excl in EXCLUDE_DIRS)]


def _compute_metrics(corpus: list[Path], constants: dict[str, float]) -> dict:
    """Compute reported metrics (never cause exit 1)."""
    from statistics import mean, stdev

    metrics = {}

    # Collect word lengths from gallery headers
    gallery_header_lengths = []
    hand_holding_stats = defaultdict(list)

    for path in corpus:
        if "plot_" in path.name and path.suffix == ".py":
            header = _extract_gallery_header(path)
            if header:
                word_count = len(header.split())
                gallery_header_lengths.append(word_count)

        if path.suffix == ".ipynb":
            cells = _extract_markdown_cells_from_ipynb(path)
            for cell_idx, cell_text, _ in cells:
                # Hand-holding: count certain pattern types
                if "the (left|right|top|bottom|lower|upper) panel" in cell_text.lower():
                    hand_holding_stats[path.name].append("panel_narration")
                if re.search(r"we\s+(build|construct|set|configure|vary|sweep)", cell_text.lower()):
                    hand_holding_stats[path.name].append("procedure")
                if re.search(r"both codes\s+(integrate|use|consume)", cell_text.lower()):
                    hand_holding_stats[path.name].append("code_reference")

    # Gallery header distribution
    if gallery_header_lengths:
        count_long = sum(1 for l in gallery_header_lengths if l >= 120)
        metrics["gallery_headers_total"] = len(gallery_header_lengths)
        metrics["gallery_headers_long"] = count_long
        metrics["gallery_headers_long_pct"] = 100.0 * count_long / len(gallery_header_lengths)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail on any violation (exit 1)")
    parser.add_argument("--report", action="store_true", help="Metrics only (no gated checks)")
    parser.add_argument("--all", action="store_true", help="Print all violations (not just first 20)")
    parser.add_argument("--fix", action="store_true", help="Rewrite notation_canon + unicode_norm in place")
    parser.add_argument("--dry-run", action="store_true", help="Preview fixes without writing (requires --fix)")
    args = parser.parse_args()

    constants = _extract_constants_from_physics()

    # Handle --fix mode
    if args.fix:
        corpus = iter_corpus()
        total_fixes = 0
        replacements_by_type = defaultdict(int)
        files_modified = []

        for path in sorted(corpus):
            # Skip generated docs/spine/*.ipynb — fix the source notebooks/*.py instead
            if "docs/spine" in path.as_posix():
                continue

            try:
                # Handle .ipynb files separately (modify markdown cells only)
                # Note: only docs/reproduction/*.ipynb are sources; docs/spine/* are generated
                if path.suffix == ".ipynb":
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        all_cells = data.get("cells", [])
                        modified = False
                        original_notebook_text = json.dumps(all_cells)  # For replacement counting

                        # Fix markdown cells only (leave code cells untouched)
                        for cell in all_cells:
                            if cell.get("cell_type") == "markdown":
                                source = cell.get("source", [])
                                if isinstance(source, list):
                                    text = "".join(source)
                                else:
                                    text = source

                                fixed = _fix_notation_canon(text)
                                fixed = _fix_unicode_normalization(fixed)

                                if fixed != text:
                                    modified = True
                                    if isinstance(source, list):
                                        cell["source"] = fixed.split('\n')
                                        # Add back newlines except after last
                                        cell["source"] = [s + '\n' for s in cell["source"][:-1]] + [cell["source"][-1]]
                                    else:
                                        cell["source"] = fixed

                        if modified:
                            total_fixes += 1
                            files_modified.append(str(path.relative_to(REPO)))

                            if not args.dry_run:
                                path.write_text(json.dumps(data, indent=1), encoding="utf-8")

                            # Count replacements by scanning for patterns in original cells
                            for cell in all_cells:
                                if cell.get("cell_type") == "markdown":
                                    source = cell.get("source", [])
                                    if isinstance(source, list):
                                        cell_text = "".join(source)
                                    else:
                                        cell_text = source

                                    # Count notation_canon replacements
                                    for canonical, variants in NOTATION_CANON.items():
                                        for variant in variants:
                                            count = len(re.findall(rf"\b{re.escape(variant)}\b", cell_text))
                                            if count > 0:
                                                replacements_by_type[f"{variant}->{canonical}"] += count

                                    # Count reference code casing
                                    for word in re.findall(r"\b[A-Za-z]+\b", cell_text):
                                        if any(word.lower() == ref.lower() for ref in REF_CODE_CANON):
                                            if word not in REF_CODE_CANON:
                                                canonical = next(ref for ref in REF_CODE_CANON if ref.lower() == word.lower())
                                                count = len(re.findall(rf"\b{re.escape(word)}\b", cell_text))
                                                if count > 0:
                                                    replacements_by_type[f"casing:{word}->{canonical}"] += count

                                    # Count unicode replacements
                                    if "µ" in cell_text:
                                        replacements_by_type["µ->μ"] += cell_text.count("µ")
                                    for ligature in ["ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ"]:
                                        if ligature in cell_text:
                                            replacements_by_type[f"ligature:{ligature}"] += cell_text.count(ligature)

                    except Exception as e:
                        print(f"Warning: failed to process {path}: {e}", file=sys.stderr)
                    continue

                # For non-ipynb files
                text = path.read_text(encoding="utf-8", errors="ignore")
                new_text = text

                # Apply fixes based on file type
                # For .py files: ONLY fix docstrings and markdown blocks, NOT executable code
                if path.suffix == ".py":
                    if "plot_" in path.name:
                        # Sphinx gallery example: fix header docstring only
                        header = _extract_gallery_header(path)
                        if header:
                            fixed_header = _fix_notation_canon(header)
                            fixed_header = _fix_unicode_normalization(fixed_header)
                            if fixed_header != header:
                                new_text = new_text.replace(header, fixed_header, 1)

                    # Fix %% [markdown] blocks (now returns tuples of (line_number, content))
                    # Must reconstruct file with fixed markdown blocks
                    blocks = _extract_percent_blocks(path)
                    if blocks:
                        lines = text.split("\n")
                        # Collect all markdown block ranges
                        markdown_ranges = []
                        for block_start_line, block_content in blocks:
                            # Find the end line of this block
                            # It ends when we encounter a non-# line or another # %% marker
                            block_end_line = block_start_line + 1
                            while block_end_line < len(lines):
                                if not lines[block_end_line].startswith("#"):
                                    break
                                if re.match(r"^#\s*%%", lines[block_end_line]):
                                    break
                                block_end_line += 1
                            markdown_ranges.append((block_start_line, block_end_line, block_content))

                        # Fix markdown content and reconstruct
                        fixed_blocks = []
                        for block_start_line, block_end_line, block_content in markdown_ranges:
                            fixed_content = _fix_notation_canon(block_content)
                            fixed_content = _fix_unicode_normalization(fixed_content)
                            fixed_blocks.append((block_start_line, block_end_line, fixed_content))

                        # Reconstruct file by replacing markdown blocks
                        if fixed_blocks:
                            new_lines = lines.copy()
                            # Process in reverse order to maintain line numbers
                            for block_start_line, block_end_line, fixed_content in reversed(fixed_blocks):
                                # Skip the header line (# %% [markdown])
                                # Replace lines[block_start_line+1:block_end_line] with fixed_content
                                fixed_content_lines = fixed_content.split("\n")
                                # Add "# " prefix to each line
                                fixed_content_lines = ["# " + line if line else "#" for line in fixed_content_lines]
                                # Replace the block
                                new_lines = new_lines[:block_start_line + 1] + fixed_content_lines + new_lines[block_end_line:]
                            new_text = "\n".join(new_lines)
                else:
                    # For markdown and RST: apply to full text (respecting code spans)
                    new_text = _fix_notation_canon(new_text)
                    new_text = _fix_unicode_normalization(new_text)

                if new_text != text:
                    # Count the specific replacements
                    # For notation_canon variants
                    for canonical, variants in NOTATION_CANON.items():
                        for variant in variants:
                            old_count = len(re.findall(rf"\b{re.escape(variant)}\b", text))
                            new_count = len(re.findall(rf"\b{re.escape(variant)}\b", new_text))
                            if old_count > new_count:
                                replacements_by_type[f"{variant}->{canonical}"] += old_count - new_count

                    # For reference code casing
                    for ref in REF_CODE_CANON:
                        for word in re.findall(r"\b[A-Za-z]+\b", text):
                            if word.lower() == ref.lower() and word != ref:
                                old_count = len(re.findall(rf"\b{re.escape(word)}\b", text))
                                new_count = len(re.findall(rf"\b{re.escape(word)}\b", new_text))
                                if old_count > new_count:
                                    replacements_by_type[f"casing:{word}->{ref}"] += old_count - new_count

                    # For unicode replacements
                    if "µ" in text and "µ" not in new_text:
                        replacements_by_type["µ->μ"] += text.count("µ")
                    for ligature in ["ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ"]:
                        if ligature in text:
                            replacements_by_type[f"ligature:{ligature}"] += text.count(ligature)

                    # For numeric ranges
                    old_ranges = len(re.findall(r"\d+-\d+", text))
                    new_ranges = len(re.findall(r"\d+-\d+", new_text))
                    if old_ranges > new_ranges:
                        replacements_by_type["numeric-range-hyphen->endash"] += old_ranges - new_ranges

                    total_fixes += 1
                    files_modified.append(str(path.relative_to(REPO)))

                    if not args.dry_run:
                        path.write_text(new_text, encoding="utf-8")

            except Exception as e:
                print(f"Warning: failed to process {path}: {e}", file=sys.stderr)

        # Report summary
        if args.dry_run:
            print(f"Dry run: would modify {total_fixes} file(s)")
        else:
            print(f"Fixed {total_fixes} file(s)")

        if replacements_by_type:
            print("\nReplacement breakdown:")
            for form, count in sorted(replacements_by_type.items()):
                print(f"  {form}: {count}")
        else:
            print("No fixes needed")

        if files_modified and (args.dry_run or total_fixes > 0):
            print(f"\nModified files ({len(files_modified)}):")
            for f in sorted(files_modified)[:20]:
                print(f"  {f}")
            if len(files_modified) > 20:
                print(f"  ... and {len(files_modified) - 20} more")

        return 0

    # Normal scan mode
    corpus = iter_corpus()
    all_violations = []

    for path in sorted(corpus):
        violations = scan_file(path, constants)
        all_violations.extend(violations)

    # Compute metrics
    metrics = _compute_metrics(corpus, constants)

    if args.report or not args.strict:
        # Print summary
        print(f"Scanned {len(corpus)} files")
        print(f"Found {len(all_violations)} prose violation(s)")
        if all_violations:
            cap = None if args.all else 20
            shown = all_violations if cap is None else all_violations[:cap]
            for v in shown:
                print(f"  {v.path}:{v.line}  [{v.check_name}] {v.message}")
            if cap and len(all_violations) > cap:
                print(f"  ... (showing {cap} of {len(all_violations)}; pass --all to show all)")

        # Print metrics
        if metrics:
            print("\nMetrics:")
            if "gallery_headers_total" in metrics:
                print(f"  Gallery headers: {metrics['gallery_headers_total']} total, "
                      f"{metrics['gallery_headers_long']} >= 120 words "
                      f"({metrics['gallery_headers_long_pct']:.1f}%)")

    if args.report:
        return 0

    if not all_violations:
        print("OK: prose passes all voice checks")
        return 0

    if args.strict:
        print(f"\nFAIL: {len(all_violations)} violation(s) found (--strict)")
        return 1

    print("\n(warn-only; pass --strict to fail the build)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
