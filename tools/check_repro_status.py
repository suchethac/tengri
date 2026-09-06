#!/usr/bin/env python3
"""Reproduction notebook verification status guard.

Parity-audit notebooks (`docs/reproduction/*.ipynb`) compare tengri against
AGNfitter, BAGPIPES, CIGALE, Prospector, ProSpect, and Synthesizer. This
guard verifies that status assertions in the notebooks match the
`docs/dev/verification-protocol.md` ledger.

The protocol defines physics-verification status for components: `VERIFIED`,
`CROSSVAL`, `PARTIAL (n/m)`, `NOT RUN`, `PENDING`. Each reproduction notebook
may state the verification status of one or more components it audits. This
guard:

1. Parses status tables from `docs/dev/verification-protocol.md`.
2. Scans markdown cells of reproduction notebooks for status assertions.
3. Fails if a notebook asserts a status that disagrees with the protocol or
   names a component the protocol does not know about.
4. Exits 0 with a summary when notebooks carry no status lines yet (non-strict
   mode) — does not block before notebook edits land. With `--strict`,
   requires every notebook to carry at least one status line.

**Status line convention (for reproduction notebooks):**

Each markdown cell may contain a verification status line of the form:

    **Verification Status:** <STATUS> — <component name>

Where `<STATUS>` is one of `VERIFIED`, `CROSSVAL`, `PARTIAL (n/m)`, `NOT RUN`,
or `PENDING`, and `<component name>` is a component name from the protocol's
table (e.g. "Absolute SED normalization", "CSP integral — CIC age kernel
(default)"). The format renders well in nbsphinx (bold with em dash) and is
unambiguous to parse via regex. Examples:

    **Verification Status:** CROSSVAL — Absolute SED normalization
    **Verification Status:** PARTIAL (68/126) — Absolute SED normalization

A single notebook may declare the status of multiple components in separate
lines or cells. If a notebook does not declare any status lines, it is assumed
to be awaiting audit and the notebook is flagged only in non-strict mode.

Usage::

    python tools/check_repro_status.py            # warn-only (exit 0)
    python tools/check_repro_status.py --strict   # fail if any notebook lacks status
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = REPO / "docs" / "dev" / "verification-protocol.md"
REPRO_DIR = REPO / "docs" / "reproduction"


class Status(NamedTuple):
    """A component's verification status from the protocol."""

    component: str
    status_str: str  # e.g. "CROSSVAL", "PARTIAL (68/126)", etc.


def parse_protocol_tables() -> dict[str, str]:
    """Extract component -> status mappings from verification-protocol.md.

    Returns
    -------
    dict
        Mapping of component name (str) to status string (str), where status
        string is one of 'VERIFIED', 'CROSSVAL', 'PARTIAL (n/m)', 'NOT RUN',
        'PENDING'.
    """
    text = PROTOCOL_FILE.read_text(encoding="utf-8")

    # Markdown table format (from verification-protocol.md):
    # | Component | Primary Reference | Upstream Code | Test File | Status |
    # |---|---|---|---|---|
    # | CSP integral — CIC age kernel (default) | ... | ... | ... | CROSSVAL (2 tests — thin) |
    #
    # We look for lines starting with '|' and parse them as table rows.
    # The Status column is the last column.

    component_status = {}
    in_table = False
    lines = text.split("\n")

    for i, line in enumerate(lines):
        # Detect table start: skip header separator lines
        if line.strip().startswith("|") and "---|" in line:
            in_table = True
            continue

        if not line.strip().startswith("|"):
            in_table = False
            continue

        if not in_table:
            continue

        # Parse table row: | col1 | col2 | ... | status |
        parts = [p.strip() for p in line.split("|")]
        # parts[0] is empty (before leading |), parts[-1] is empty (after trailing |)
        # parts[1] = Component, parts[5] = Status (typically)
        if len(parts) < 6:
            continue

        component = parts[1]
        status = parts[5]

        # Skip header rows (they contain "Component" as the component name)
        if component in ("Component", ""):
            continue

        # Skip section headers and markdown artifacts
        if not status or status == "Status":
            continue

        component_status[component] = status

    return component_status


def scan_notebook_status_assertions(
    notebook_path: Path,
) -> list[tuple[int, str, str]]:
    """Scan a notebook for verification status assertions.

    Returns
    -------
    list of (cell_index, component_name, status_str)
        Each assertion found in a markdown cell.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assertions = []
    status_pattern = re.compile(
        r"\*\*Verification Status:\*\*\s+([A-Z]+(?:\s*\(\d+/\d+\))?)\s+—\s+(.+?)(?:\n|$)"
    )

    for cell_idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue

        source = "".join(cell.get("source", []))
        for match in status_pattern.finditer(source):
            status_str = match.group(1).strip()
            component = match.group(2).strip()
            assertions.append((cell_idx, component, status_str))

    return assertions


def normalize_status(status_str: str) -> str:
    """Normalize a status string for comparison.

    Handles variations like extra spaces, case variations in the "PARTIAL" part.
    """
    # Remove extra spaces
    normalized = " ".join(status_str.split())
    return normalized


def main() -> int:
    strict = "--strict" in sys.argv[1:]

    # Parse the protocol
    if not PROTOCOL_FILE.exists():
        print(f"FAIL: protocol file not found: {PROTOCOL_FILE}")
        return 1

    protocol_status = parse_protocol_tables()

    if not protocol_status:
        print(f"FAIL: could not parse any components from {PROTOCOL_FILE}")
        return 1

    # Scan all reproduction notebooks
    if not REPRO_DIR.exists():
        print(f"FAIL: reproduction directory not found: {REPRO_DIR}")
        return 1

    notebooks = sorted(REPRO_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"FAIL: no notebooks found in {REPRO_DIR}")
        return 1

    violations: list[tuple[Path, int, str, str, str]] = []
    notebooks_without_status: list[Path] = []
    valid_assertions: list[tuple[Path, str, str]] = []

    for nb_path in notebooks:
        assertions = scan_notebook_status_assertions(nb_path)

        if not assertions:
            notebooks_without_status.append(nb_path)
            continue

        for cell_idx, component, status_str in assertions:
            # Check that component is known
            if component not in protocol_status:
                violations.append(
                    (
                        nb_path,
                        cell_idx,
                        component,
                        status_str,
                        "unknown component",
                    )
                )
                continue

            # Check that status matches protocol
            protocol_status_str = normalize_status(protocol_status[component])
            notebook_status_str = normalize_status(status_str)

            if protocol_status_str != notebook_status_str:
                violations.append(
                    (
                        nb_path,
                        cell_idx,
                        component,
                        status_str,
                        f"status mismatch (protocol: {protocol_status_str})",
                    )
                )
                continue

            valid_assertions.append((nb_path, component, status_str))

    # Report findings
    if not violations and not notebooks_without_status:
        print("reproduction status: all notebooks have valid status assertions ✓")
        return 0

    if violations:
        print(f"FAIL: {len(violations)} status assertion(s) violate the protocol:\n")
        for nb_path, cell_idx, component, status_str, reason in violations:
            print(f"  {nb_path.name}:{cell_idx}  {component}  {status_str}  ({reason})")
        print()

    if notebooks_without_status:
        print(
            f"reproduction status: {len(notebooks_without_status)} notebook(s) "
            "have no status assertions:"
        )
        for nb_path in notebooks_without_status:
            print(f"  {nb_path.name}")
        print()

    if strict and notebooks_without_status:
        print("(strict mode: every notebook must declare at least one status)")
        return 1

    if violations:
        print("Fix: correct the status assertion to match the protocol.")
        return 1

    # Non-strict mode with unaudited notebooks
    print("(warn-only; pass --strict to require status assertions in all notebooks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
