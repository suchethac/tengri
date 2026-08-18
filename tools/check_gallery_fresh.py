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

This script also detects when committed gallery pages have lost their output
artifacts: if a ``.rst`` file has corresponding images (``sphx_glr_<name>_001.png``)
committed to the repository but the ``.rst`` file no longer references them,
the output has been stripped. This can happen with an incomplete ``make html``
that skips rendering but still rewrites the ``.rst`` files.

Usage::

    python tools/check_gallery_fresh.py            # warn-only (exit 0)
    python tools/check_gallery_fresh.py --strict   # fail on any drift or stripped output
"""

from __future__ import annotations

import hashlib
import re
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


def _rst_references_images(rst_path: Path) -> bool:
    """Check if the .rst file references any output images.

    Parameters
    ----------
    rst_path : Path
        Path to the .rst file.

    Returns
    -------
    bool
        True if the file contains image directives (.. image-sg:: or .. image::),
        False otherwise.
    """
    if not rst_path.is_file():
        return False
    content = rst_path.read_text()
    # Check for both image-sg (sphinx-gallery) and image directives
    return bool(re.search(r"^\s*\.\.\s+(image-sg|image)::", content, re.MULTILINE))


def stripped_output_examples() -> list[str]:
    """Return example paths whose committed images are not referenced in .rst.

    An example is considered to have stripped output if:
    - It has a .py.md5 stamp (meaning it's tracked as rendered)
    - It has corresponding image files (sphx_glr_<name>_*.png)
    - But the .rst file doesn't reference them (no image directives)

    This detects when ``make html`` skipped rendering but still rewrote the
    .rst file, leaving orphaned image artifacts (#1236).

    Returns
    -------
    list[str]
        List of example paths (relative to ``examples/``) with stripped output.
    """
    stripped: list[str] = []
    for src in sorted(EXAMPLES.rglob("plot_*.py")):
        rel = src.relative_to(EXAMPLES)
        stem = rel.stem  # e.g., "plot_igm_redshift"
        section = rel.parent  # e.g., "igm"

        # Only check examples with a committed stamp (tracked as rendered)
        stamp = AUTO / section / (rel.name + ".md5")
        if not stamp.is_file():
            continue

        # Check for any image files in the images/ directory
        images_dir = AUTO / section / "images"
        if not images_dir.is_dir():
            continue

        # Look for sphx_glr_<stem>_*.png (output images from this example)
        image_files = list(images_dir.glob(f"sphx_glr_{stem}_*.png"))
        if not image_files:
            continue

        # If images exist, the .rst file must reference them
        rst_path = AUTO / section / (rel.name.replace(".py", ".rst"))
        if not _rst_references_images(rst_path):
            stripped.append(str(rel))

    return stripped


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    stale, unrendered = stale_examples()
    stripped = stripped_output_examples()

    if not stale and not unrendered and not stripped:
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
    if stripped:
        _show("have committed images but stripped .rst output", stripped)

    all_issues = (*stale, *unrendered, *stripped)
    names = " ".join(sorted({Path(p).stem for p in all_issues}))
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
