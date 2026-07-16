# SPDX-License-Identifier: BSD-3-Clause
"""Guard test: strict inventory of raw flux-scale patterns.

This test walks the src/tengri/ tree and enforces that the RUNTIME use of the
(1+z)/(4π d_L²) or 4π d_L² patterns appears ONLY in the documented allow-list,
and that all allow-listed files still contain their expected patterns.

Allowed exceptions (Tier B deferred — stored-scale/table sites):
- utils/grid_interp.py — stores flux_scale into PreintegratedGrid
- components/stellar/sps/precompute.py — numpy host build-time table sites
- measure.py — returns 4π d_L² as a scalar (caller-side fix)
- components/nebular/line_precompute.py — precompute site
- forward/sed_model.py — immediate-application property/flux projection
- observation/observation.py — immediate-application observation flux projection

The test is a two-way gate: NEW unauthorized sites are rejected, and STALE
allow-list entries (files that no longer contain the pattern) are flagged as
needing removal.

See issue #1186.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path("src/tengri")

# Controller-authorized allow-list: each file mapped to its rationale.
# When a file is converted (Task 6), remove its entry and its rationale.
ALLOW = {
    "utils/grid_interp.py": (
        "Tier B — stores flux_scale scalar into PreintegratedGrid, applied elsewhere; "
        "needs log-table threading"
    ),
    "components/stellar/sps/precompute.py": (
        "Tier B — stores flux_scale into ztable, applied at multiply sites; "
        "needs log-table threading"
    ),
    "measure.py": (
        "Tier B — returns 4pi dL^2 (~1e57), f32-unrepresentable as a scalar; caller-side fix"
    ),
    "components/nebular/line_precompute.py": "Tier B — returns 4pi dL^2 scalar; caller-side fix",
    "forward/sed_model.py": (
        "Tier B — line 4536 returns 4pi dL^2 scalar (f32-unrepresentable); "
        "immediate-application sites converted in Task 6"
    ),
}

# Match raw flux-scale patterns: dl_cm**2 or 4.0 * jnp.pi * dl_cm (with optional asarray wrapper)
PATTERN = re.compile(
    r"dl_cm\s*\*\*\s*2|4\.0\s*\*\s*(?:jnp\.pi|_math\.pi|np\.pi)\s*\*\s*(?:jnp\.asarray\()?dl_cm"
)


def test_no_raw_runtime_flux_scale():
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
        f"raw flux-scale sites must be in ALLOW list with documented rationale: {offenders}"
    )

    # 2. Detect stale allow-list entries (file no longer contains the pattern)
    stale = set(ALLOW.keys()) - found_files
    assert not stale, (
        f"stale allow-list entries (files no longer containing the pattern): {stale}. "
        f"If Task 6 converted them, remove the entry and rationale from ALLOW dict."
    )
