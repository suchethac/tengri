# SPDX-License-Identifier: BSD-3-Clause
"""Guard test: new raw flux-scale patterns are rejected.

This test walks the src/tengri/ tree and flags any RUNTIME use of the
(1+z)/(4π d_L²) or 4π d_L² patterns that should be routed through
utils.scale.LOG10_4PI and apply_log10_scale() instead.

Allowed exceptions (Tier B deferred — stored-scale/table sites):
- utils/grid_interp.py:495 — stores flux_scale into PreintegratedGrid
- components/stellar/sps/precompute.py:458, :958 — numpy host build-time
- measure.py:157 — returns 4π d_L² as a scalar (caller-side fix Tier B)

See issue #1186.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path("src/tengri")

# Tier-B deferred: stored/table scales applied elsewhere, or higher-level sites
ALLOW = {
    "utils/grid_interp.py",  # :495 — stored flux_scale in PreintegratedGrid
    "components/stellar/sps/precompute.py",  # :458 (jax), :958 (numpy) — table sites
    "measure.py",  # :157 — returns 4π d_L² scalar (caller fix T-B)
    "forward/sed_model.py",  # :4011, :4200, :4534 — higher-level sites (Tier B)
    "observation/observation.py",  # :1137, :1309 — higher-level sites (Tier B)
    "components/nebular/line_precompute.py",  # :89 — precompute site (Tier B)
}

# Match raw flux-scale patterns: dl_cm**2 or 4.0 * jnp.pi * dl_cm
PATTERN = re.compile(r"dl_cm\s*\*\*\s*2|4\.0\s*\*\s*(?:jnp\.pi|_math\.pi|np\.pi)\s*\*\s*dl_cm")


def test_no_raw_runtime_flux_scale():
    """Flux-scale must use utils.scale helpers, not raw (1+z)/(4π d_L²) arithmetic."""
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if rel in ALLOW:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if PATTERN.search(line):
                offenders.append(f"{rel}:{i}")
    assert not offenders, (
        f"raw flux-scale sites must use utils.scale (LOG10_4PI, apply_log10_scale): {offenders}"
    )
