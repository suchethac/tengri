# SPDX-License-Identifier: BSD-3-Clause
"""Guard test: strict inventory of raw nion-reader patterns.

This test walks the src/tengri/ tree and enforces that the RUNTIME use of
`derived["nion"]` patterns appears ONLY in the documented allow-list, and
that all allow-listed files still contain their expected patterns.

**ALLOW is now empty (#1206 §C).** The Tier B item 3 migration (retire linear
``q_h``) has landed everywhere this guard once carved out an exception:

- ``components/stellar/component.py`` — the linear ``_q_h_fn`` / ``Property("q_h")``
  is removed with no alias; ``log_q_h`` is the sole surviving form.
- ``forward/component_factory.py`` — ``state_to_ionizing_quantities`` no longer
  reads ``derived["nion"]`` at all; its ``IonizingQuantities`` NamedTuple dropped
  the ``q_h`` field.
- ``components/nebular/nebular_grid_precompute.py`` — ``_log_nion_of_state``'s
  fallback to ``jnp.log10(state.derived["nion"])`` (for a hypothetical state
  carrying only the linear publish) is removed: ``log_nion`` is published in the
  same ``apply()`` call as ``nion`` on every real model, so the branch was
  unreachable and was this guard's last allow-listed site.

``forward/sed_model.py`` and ``components/nebular/line_precompute.py`` were
removed from this list earlier, by the #1206 line-flux float32 fix: see git
history for that rationale.

The test is a two-way gate: NEW unauthorized sites are rejected, and STALE
allow-list entries (files that no longer contain the pattern) are flagged as
needing removal. An empty ALLOW makes the second half vacuous by construction,
but the check stays: a future Tier-B-shaped exception should re-populate it
explicitly, not silently widen the pattern.

See issue #1206.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path("src/tengri")

# Controller-authorized allow-list: each file mapped to its rationale.
# Empty (#1206 §C): every previously-deferred site has migrated. Re-populate
# with a documented rationale if a new one is ever genuinely needed.
ALLOW: dict[str, str] = {}

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
