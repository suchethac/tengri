# SPDX-License-Identifier: BSD-3-Clause
"""Guard test: strict inventory of raw nion-reader patterns.

This test walks the src/tengri/ tree and enforces that the RUNTIME use of
`derived["nion"]` patterns appears ONLY in the documented allow-list, and
that all allow-listed files still contain their expected patterns.

Allowed exceptions (Tier B deferred — linear q_h / erg/s paths):
- components/stellar/component.py — _q_h_fn linear q_h property (photons/s ~1e56)
- forward/component_factory.py — state_to_ionizing_quantities.q_h linear surface
- components/nebular/line_precompute.py — build-path _nion_of_state (erg/s + 4pi dL^2)
- components/nebular/nebular_grid_precompute.py — build-path _nion_of_state (f64 build)

``forward/sed_model.py`` was on this list and is not any more: its fast-line path
took the linear ``nion`` (~1e53, ``inf`` in float32) only to feed an erg/s line
reconstruct that then overflowed too, which was the whole of why a float32 Cue line
fit could not run. It now reads ``log_nion`` and stays in the exponent, so the
Tier B item 3 migration has landed for that file and its entry is removed rather
than kept green by widening the pattern.

Each entry will be removed when its corresponding Tier B item (2 or 3) migration lands
(see issue #1206).

The test is a two-way gate: NEW unauthorized sites are rejected, and STALE
allow-list entries (files that no longer contain the pattern) are flagged as
needing removal.

See issue #1206.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path("src/tengri")

# Controller-authorized allow-list: each file mapped to its rationale.
# When a file is converted (Tier B item 2/3 migration), remove its entry and rationale.
ALLOW = {
    "components/stellar/component.py": "Tier B item 3 — _q_h_fn linear q_h property (~1e56)",
    "forward/component_factory.py": "Tier B item 3 — state_to_ionizing_quantities.q_h linear",
    "components/nebular/line_precompute.py": "Tier B item 2/3 — build-path _nion_of_state (erg/s)",
    "components/nebular/nebular_grid_precompute.py": "Tier B item 2/3 — build _nion_of_state",
}

# Match raw nion-reader patterns: derived["nion"] or derived.get("nion"
PATTERN = re.compile(r'derived\s*\[\s*"nion"\s*\]|derived\.get\(\s*"nion"')


def test_no_raw_nion_read():
    """Two-way gate: reject NEW sites and detect stale allow-list entries."""
    # Inventory: collect all files that match the pattern
    offenders = []
    found_files = set()
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        has_match = False
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if PATTERN.search(line):
                has_match = True
                # Only flag if NOT in allow-list
                if rel not in ALLOW:
                    offenders.append(f"{rel}:{i}")
        if has_match:
            found_files.add(rel)

    # Two-way gate checks:
    # 1. Reject NEW unauthorized sites
    assert not offenders, (
        f"raw nion-read sites must be in ALLOW list with documented rationale: {offenders}"
    )

    # 2. Detect stale allow-list entries (file no longer contains the pattern)
    stale = set(ALLOW.keys()) - found_files
    assert not stale, (
        f"stale allow-list entries (files no longer containing the pattern): {stale}. "
        f"If a Tier B item 2/3 migration converted them, remove the entry from ALLOW."
    )
