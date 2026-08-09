#!/usr/bin/env python
"""Gallery freshness guard — flag examples whose committed render is stale.

Sphinx-gallery records the md5 of each example source it rendered in
``docs/auto_examples/<section>/<name>.py.md5``. When an example's source
changes but its committed render is not refreshed, that stamp drifts from the
current source — the docs site then shows a figure/code that no longer matches
the example. This script compares each ``examples/**/plot_*.py`` to its stamp
and reports the drift.

CI runs this with ``--strict`` (#805). It ran warn-only until 2026-08, during
which the drift grew from 16 examples to 60 — nobody acts on a report that
cannot fail. The earlier reason for warn-only, that heavy VI/MCMC/NUTS
examples must keep their committed figures and would drift forever, no longer
applies: the 2026-07 overhaul removed them, and every remaining fit example is
MAP or native-VI (see ``docs/conf.py``).

Fixing a drift means re-rendering the example, which CI cannot do — it has
neither the SSP grids nor the ~20 GB of optional data. It belongs to whoever
edited the source::

    python tools/regen_gallery.py <basename>

Not a bare ``make html``: that rewrites every page while executing almost
none, so the ones it skips come back stripped of their output (#1236).

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
    names = " ".join(sorted({Path(p).stem for p in (*stale, *unrendered)}))
    print(
        f"\nRefresh with:\n    python tools/regen_gallery.py {names}\n"
        "then commit the regenerated docs/auto_examples/ files.\n"
        "Do NOT use a bare `make html`: it rewrites every example page but "
        "executes almost none, so the skipped ones lose their output (#1236)."
    )

    if strict:
        return 1
    print("\n(warn-only; pass --strict to fail the build)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
