#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Guard notebook renders against dead fits and diagnostic failures.

A committed notebook render that shows a dead fit or divergent inference must
fail this guard. The rendered output carries printed diagnostics — R-hat values,
NaN parameters, log_evidence warnings — that indicate the fit is unreliable.

Violations
----------

A notebook cell output fails the guard if it contains:

1. **Dead fit markers**: ``DeadFitWarning`` or ``DeadFitError`` text indicates
   some parameter(s) have only one unique draw out of hundreds of samples (#1999).
   The fit is unusable.

2. **Catastrophic R-hat values**: Printed R-hat values in the range (2.0, ∞) or
   non-finite (NaN/Inf) indicate divergent or frozen sampling. The threshold is
   deliberately high (2.0, not 1.01) to catch only the frozen regime (~1e13–1e14
   on collapsed fits). Post-#2110, divergent-dead fails at execution time; frozen-dead
   still renders (tracked as #2112 for per-chain spread detection). Honest comparison
   arm in nb06 prints R-hat 1.271 which passes. Patterns matched:
   - ``max R̂ = <float>``
   - ``R̂ = <float>`` (also matches ``max split-R̂ = <float>``, measured in quickstart cell 11)
   - ``max R-hat <float>``
   - ``R-hat <float>``

3. **NaN in parameter summary tables**: A printed row of posterior parameter
   summaries that contains ``nan``/``NaN`` as a VALUE (not in prose or base64
   blobs). The scan is precise to table-shaped output lines and excludes image
   payloads.

4. **Unhandled log_evidence NaN**: A line printing ``log_evidence: nan`` without
   an explicit fallback. A line that both reports the NaN AND states a fallback
   (e.g., "⚠ ... using NSS weights") passes with a WARN note, not a failure.
   Tracked as #1985 (the BMA render's handled case).

Known-bad ledger
----------------

Entries in this guard are bug reports to open issues. Removing a ledger entry
implies fixing the underlying problem. Every entry must cite an open issue by
number.

.. code-block:: python

    KNOWN_BAD_LEDGER = {
        # Dead fits (seed 7 marginal-to-dead at both code endpoints): #2095
        "notebooks/05_fitting_photometry.ipynb": "#2095",
        "docs/spine/05_fitting_photometry.ipynb": "#2095",
        # Vacuous DeadFitWarnings on Fixed parameters (pre-#2090 bug): #2113
        "notebooks/00_quickstart.ipynb": "#2113",
        "notebooks/01_why_jax.ipynb": "#2113",
        "notebooks/06_fitting_spectroscopy.ipynb": "#2113",
        "notebooks/07_joint_photo_spec.ipynb": "#2113",
        "notebooks/11_catalog_fits.ipynb": "#2113",
        "docs/spine/00_quickstart.ipynb": "#2113",
        "docs/spine/01_why_jax.ipynb": "#2113",
        "docs/spine/06_fitting_spectroscopy.ipynb": "#2113",
        "docs/spine/07_joint_photo_spec.ipynb": "#2113",
        "docs/spine/11_catalog_fits.ipynb": "#2113",
        "docs/spine/experimental/jwst_nonparametric_fits.ipynb": "#2113",
        "docs/spine/experimental/stochastic_sfh_recovery.ipynb": "#2113",
    }

**#2095** — ``05_fitting_photometry.ipynb`` cannot be re-rendered healthy right
now (seed 7 is marginal-to-dead at both code endpoints); a separate decision
gates its re-render.

**#2113** — The other eleven notebooks carry vacuous DeadFitWarnings that fired
on Fixed parameters with healthy fits (pre-#2090 bug in the warning logic).
Re-rendering will remove these warnings automatically.

Usage
-----

::

    python tools/check_render_diagnostics.py
    python tools/check_render_diagnostics.py --list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Known bad notebooks cited to specific issues. Ledger entries are bug reports;
# removing the entry implies fixing the issue. Entries whose files no longer exist
# on disk are silently skipped (a deleted notebook should not fail the guard).
# Removing a notebook from the repository also requires manual removal of its entry.
KNOWN_BAD_LEDGER = {
    # Dead fits (seed 7 marginal-to-dead at both code endpoints): #2095
    "notebooks/05_fitting_photometry.ipynb": "#2095",
    "docs/spine/05_fitting_photometry.ipynb": "#2095",
    # Vacuous DeadFitWarnings on Fixed parameters (pre-#2090 bug): #2113
    "notebooks/00_quickstart.ipynb": "#2113",
    "notebooks/01_why_jax.ipynb": "#2113",
    "notebooks/06_fitting_spectroscopy.ipynb": "#2113",
    "notebooks/07_joint_photo_spec.ipynb": "#2113",
    "notebooks/11_catalog_fits.ipynb": "#2113",
    "docs/spine/00_quickstart.ipynb": "#2113",
    "docs/spine/01_why_jax.ipynb": "#2113",
    "docs/spine/06_fitting_spectroscopy.ipynb": "#2113",
    "docs/spine/07_joint_photo_spec.ipynb": "#2113",
    "docs/spine/11_catalog_fits.ipynb": "#2113",
    "docs/spine/experimental/jwst_nonparametric_fits.ipynb": "#2113",
    "docs/spine/experimental/stochastic_sfh_recovery.ipynb": "#2113",
}


def _collect_notebook_paths() -> list[Path]:
    """Recursively find all .ipynb files under notebooks/ and docs/spine/."""
    paths = []
    for root_dir in ["notebooks", "docs/spine"]:
        base = ROOT / root_dir
        if base.exists():
            paths.extend(sorted(base.glob("**/*.ipynb")))
    return paths


def _get_relative_path(path: Path) -> str:
    """Path relative to repo root, for logging."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _extract_text_from_output(output: dict) -> str:
    """Extract all text content from a notebook cell output."""
    text = ""

    # Text outputs
    if "text" in output:
        t = output["text"]
        text += "".join(t) if isinstance(t, list) else str(t)

    # Traceback outputs (if present)
    if "traceback" in output:
        text += "\n".join(output["traceback"])

    # Data fields (text/plain, text/html, etc.)
    data = output.get("data") or {}
    for key in ["text/plain", "text/html"]:
        if key in data:
            v = data[key]
            text += "".join(v) if isinstance(v, list) else str(v)

    return text


def _check_dead_fit_warning(text: str) -> bool:
    """Check for DeadFitWarning or DeadFitError text."""
    return "DeadFitWarning" in text or "DeadFitError" in text


def _check_catastrophic_rhat(text: str) -> bool:
    """Check for R-hat values in catastrophic regime (>2.0 or non-finite).

    Patterns:
    - max R̂ = <float>
    - R̂ = <float>
    - max R-hat <float>
    - R-hat <float>
    """
    # Patterns for R-hat values
    patterns = [
        r"max R̂\s*=\s*([\d\.\-eE\+]+)",
        r"R̂\s*=\s*([\d\.\-eE\+]+)",
        r"max R-hat\s+([\d\.\-eE\+]+)",
        r"R-hat\s+([\d\.\-eE\+]+)",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                value = float(match.group(1))
                # Non-finite or > 2.0 is catastrophic
                if not (-float("inf") < value < float("inf")) or value > 2.0:
                    return True
            except (ValueError, IndexError):
                pass
    return False


def _check_nan_in_table(text: str) -> bool:
    """Check for nan/NaN as a VALUE in posterior parameter-summary table rows.

    Parameter summary tables typically have:
    - Leading whitespace (indentation)
    - Parameter name (identifier with underscores like sfh_dpl_beta)
    - Multiple numeric values
    - Possibly nan/NaN as one of the values

    This distinguishes them from prose which typically starts with words,
    not indented parameter names.
    """
    for line in text.split("\n"):
        if not line or not line[0].isspace():
            # Prose and table headers typically don't start with indent,
            # but parameter value rows do
            continue
        # Check if this looks like a parameter name followed by values
        # Pattern: leading space(s), then parameter-like name (word_word),
        # then multiple numeric/nan values
        match = re.match(r"\s+[\w_]+\s+", line)
        if not match:
            continue
        # Now check if nan/NaN appears in the values part after the parameter name
        values_part = line[match.end() :]
        if re.search(r"\s(nan|NaN)\s", " " + values_part + " "):
            # Also confirm there are other numeric values
            numeric_values = re.findall(r"[\d\-\+\.eE]+", values_part)
            if len(numeric_values) >= 1:  # At least 1 numeric value + nan
                return True
    return False


def _check_unhandled_log_evidence_nan(text: str) -> tuple[bool, bool]:
    """Check for log_evidence: nan.

    Returns (failed, warned):
    - failed=True if there's an unhandled log_evidence: nan
    - warned=True if there's a handled one (with fallback stated)
    """
    if "log_evidence: nan" not in text and "log_evidence: NaN" not in text:
        return False, False

    # Look for the pattern with a fallback
    handled_pattern = r"log_evidence:\s*(?:nan|NaN).*(?:using|fallback|NSS)"
    if re.search(handled_pattern, text, re.IGNORECASE):
        return False, True  # Handled, warn but don't fail
    return True, False  # Unhandled, fail


def check_notebook(path: Path) -> tuple[list[str], list[str]]:
    """Check a notebook for violations.

    Returns (failures, warnings). Empty lists mean the notebook is clean.
    """
    failures = []
    warnings = []

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"unreadable notebook: {exc}")
        return failures, warnings

    for cell_idx, cell in enumerate(nb.get("cells", [])):
        for output in cell.get("outputs", []):
            text = _extract_text_from_output(output)

            # Check for dead fit warnings
            if _check_dead_fit_warning(text):
                failures.append(f"cell {cell_idx}: DeadFitWarning detected")

            # Check for catastrophic R-hat
            if _check_catastrophic_rhat(text):
                failures.append(f"cell {cell_idx}: R-hat value > 2.0 or non-finite")

            # Check for NaN in tables
            if _check_nan_in_table(text):
                failures.append(f"cell {cell_idx}: NaN in parameter summary table")

            # Check for log_evidence: nan
            failed, warned = _check_unhandled_log_evidence_nan(text)
            if failed:
                failures.append(f"cell {cell_idx}: unhandled log_evidence: nan")
            if warned:
                warnings.append(f"cell {cell_idx}: handled log_evidence: nan (with fallback)")

    return failures, warnings


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print every notebook checked")
    args = ap.parse_args(argv)

    notebooks = _collect_notebook_paths()
    real_problems: list[str] = []  # Real failures (not in ledger)
    known_bad: list[str] = []  # Known-bad entries (tracked issues)
    stale_entries: list[str] = []  # Ledgered files with zero failures (must be removed)
    warnings: list[str] = []
    checked_ledger_entries = set()

    for path in notebooks:
        rel_path = _get_relative_path(path)
        if args.list:
            print(f"  {rel_path}")
            continue

        failures, warns = check_notebook(path)

        # Check if this notebook is in the known-bad ledger
        if rel_path in KNOWN_BAD_LEDGER:
            checked_ledger_entries.add(rel_path)
            if failures:
                issue = KNOWN_BAD_LEDGER[rel_path]
                known_bad.append(f"{rel_path}: KNOWN-BAD ({issue})")
            else:
                # Ledgered file has zero failures: the bug is fixed, remove the entry
                issue = KNOWN_BAD_LEDGER[rel_path]
                stale_entries.append(f"{rel_path}: stale ledger entry — remove it (cites {issue})")
            continue

        # Report real failures (not in ledger)
        for failure in failures:
            real_problems.append(f"{rel_path}: {failure}")

        # Collect warnings
        for warn in warns:
            warnings.append(f"{rel_path}: {warn}")

    if args.list:
        return 0

    # Report results: fail if there are real problems or stale ledger entries
    all_issues = stale_entries + known_bad + real_problems

    if stale_entries or real_problems:
        # There are actual failures or stale ledger entries
        num_issues = len(stale_entries) + len(real_problems)
        print(f"FAIL: {num_issues} notebook(s) have issues:\n", file=sys.stderr)
        for p in all_issues:
            print(f"  {p}", file=sys.stderr)
        return 1

    # Known-bad entries are reported but don't fail the run
    msg = f"OK: {len(notebooks)} notebook(s) checked"
    if known_bad:
        msg += f"; {len(known_bad)} KNOWN-BAD"
        for p in known_bad:
            print(f"  {p}")
    if warnings:
        msg += f"; {len(warnings)} warning(s)"
    print(msg + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
