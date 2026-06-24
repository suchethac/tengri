#!/usr/bin/env python
"""Gallery freshness guard — flag examples whose committed render is stale.

Sphinx-gallery records the md5 of each example source it rendered in
``docs/auto_examples/<section>/<name>.py.md5``. When an example's source
changes but its committed render is not refreshed, that stamp drifts from the
current source — the docs site then shows a figure/code that no longer matches
the example. This script compares each ``examples/**/plot_*.py`` to its stamp
and reports the drift.

Default mode is **warn-only** (exit 0): it prints the stale list so a human /
CI annotation can see it, without failing the build. Heavy fit examples
(VI/MCMC/NUTS) are intentionally left on their committed figures (re-executing
them stalls/OOMs the doc build), so their stamps are expected to drift — a hard
failure here would be a false alarm. Pass ``--strict`` to exit non-zero on any
drift once the heavy set is handled (see #612 / #805).

Usage::

    python tools/check_gallery_fresh.py            # warn-only (exit 0)
    python tools/check_gallery_fresh.py --strict   # fail on any drift
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
AUTO = REPO / "docs" / "auto_examples"


def stale_examples() -> tuple[list[str], list[str]]:
    """Return (stale, unrendered) example paths relative to ``examples/``."""
    stale: list[str] = []
    unrendered: list[str] = []
    for src in sorted(EXAMPLES.rglob("plot_*.py")):
        rel = src.relative_to(EXAMPLES)
        stamp = AUTO / rel.parent / (rel.name + ".md5")
        if not stamp.is_file():
            unrendered.append(str(rel))
            continue
        current = hashlib.md5(src.read_bytes()).hexdigest()
        if current != stamp.read_text().strip():
            stale.append(str(rel))
    return stale, unrendered


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    stale, unrendered = stale_examples()

    if not stale and not unrendered:
        print("gallery freshness: all examples match their committed render ✓")
        return 0

    cap = 40

    def _show(label: str, items: list[str]) -> None:
        print(f"gallery freshness: {len(items)} example(s) {label}:")
        for x in items[:cap]:
            print(f"  {x}")
        if len(items) > cap:
            print(f"  ... and {len(items) - cap} more")

    if stale:
        _show("drifted from their committed render", stale)
    if unrendered:
        _show("have no committed render", unrendered)
    print(
        "\nRefresh by deleting the example's "
        "docs/auto_examples/<section>/images/sphx_glr_<name>_001.png and "
        "rebuilding the gallery (make html), then commit the regenerated files."
    )

    if strict:
        return 1
    print("\n(warn-only; pass --strict to fail the build)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
