# SPDX-License-Identifier: BSD-3-Clause
"""Guard test: strict inventory of raw flux-scale patterns.

This test walks the src/tengri/ tree and enforces that the RUNTIME use of the
(1+z)/(4π d_L²) or 4π d_L² patterns appears ONLY in the documented allow-list,
and that all allow-listed files still contain their expected patterns.

**The allow-list is now empty, and that is the point.** It carried five entries
from Tier A (#1186), each deferring a site that *stored* or *passed* the scale as
a standalone scalar rather than applying it immediately — ``PreintegratedGrid``,
the spectroscopic and photometric z-tables, and the three producers of the line
divisor. #1859 measured what the deferral cost in float32 and converted them:

- the line path was ``nan`` at every redshift (``inf/inf``: the erg/s line
  luminosity ~1.4e40 over ``4 pi d_L^2`` ~1.0e57, both outside float32);
- the stored photometry scales were exactly ``0.0``, which is the dangerous
  half — finite, sign-correct, and wrong by every order of magnitude at once.

Every site now goes through :func:`tengri.utils.scale.log10_flux_scale` or
:func:`~tengri.utils.scale.log10_four_pi_dl2` and is applied with
:func:`~tengri.utils.scale.apply_log10_scale`, so this guard's job from here is
purely to stop a thirteenth hand-written copy appearing. Twelve existed before
(seven correct, five not); one named helper replaced all of them.

The test is a two-way gate: NEW unauthorized sites are rejected, and STALE
allow-list entries (files that no longer contain the pattern) are flagged as
needing removal. An empty ALLOW makes the second half vacuous and the first half
absolute — which is the intended end state, not an oversight.

See issues #1186 and #1859.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path("src/tengri")

# Controller-authorized allow-list: each file mapped to its rationale.
#
# EMPTY BY DESIGN (#1859). The five Tier-B deferrals it used to hold were all
# converted together; leaving any of them here after the conversion would be a
# lie the two-way gate is specifically built to catch. Add an entry only with a
# rationale that says why the scale cannot be carried as a log10 offset — and
# note that "I need the linear value" is not such a reason, because there is no
# distance at which the linear value is representable in float32.
ALLOW: dict[str, str] = {}

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
