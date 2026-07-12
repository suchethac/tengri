#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Generate a markdown table of all registered properties.

This script introspects the property registry and writes a deterministically
ordered markdown table to `docs/api/_property_table.md`. The table is used
in documentation to list all available derived quantities.

Usage
-----
Generate or update the table::

    python tools/gen_property_table.py

Check if the table is up to date (for CI)::

    python tools/gen_property_table.py --check
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import tengri
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tengri.forward.properties import PROPERTY_REGISTRY


def generate_table_content() -> str:
    """Generate markdown table content from the property registry.

    Returns
    -------
    str
        Markdown-formatted table with columns: name, units, group, component, description.
    """
    if not PROPERTY_REGISTRY:
        return "No properties registered.\n"

    # Collect all unique (name, entry) pairs, sorted for determinism
    rows = []
    for name in sorted(PROPERTY_REGISTRY.keys()):
        entries = PROPERTY_REGISTRY[name]
        # Use first entry (most models have only one per name; collisions raise at build time)
        entry = entries[0]
        rows.append(
            (
                name,
                entry.units if entry.units else "—",
                entry.group,
                entry.component_name,
                entry.doc,
            )
        )

    # Build markdown table
    lines = [
        "| Name | Units | Group | Component | Description |",
        "|------|-------|-------|-----------|-------------|",
    ]
    for name, units, group, component, doc in rows:
        # Escape pipe characters in description
        doc_safe = doc.replace("|", "\\|") if "|" in doc else doc
        lines.append(f"| `{name}` | {units} | {group} | {component} | {doc_safe} |")

    return "\n".join(lines) + "\n"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate property table for tengri documentation."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the table is up to date; exit 1 if stale.",
    )
    args = parser.parse_args()

    # Output file path
    output_file = Path(__file__).parent.parent / "docs" / "api" / "_property_table.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate new content
    new_content = generate_table_content()

    if args.check:
        # Check mode: compare against committed file
        if output_file.exists():
            existing_content = output_file.read_text()
            if existing_content == new_content:
                print(f"✓ Property table is up to date: {output_file}")
                return 0
            else:
                print(
                    f"✗ Property table is stale: {output_file}\n"
                    f"  Run: python tools/gen_property_table.py"
                )
                return 1
        else:
            print(
                f"✗ Property table does not exist: {output_file}\n"
                f"  Run: python tools/gen_property_table.py"
            )
            return 1
    else:
        # Generate mode: write the table
        output_file.write_text(new_content)
        print(f"✓ Generated property table: {output_file}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
