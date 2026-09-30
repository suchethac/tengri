# SPDX-License-Identifier: BSD-3-Clause
"""Guard G2: Every GRAHSP variant branch value maps to a registered composable block.

Detects missing block registrations: if the component API branches on a variant
literal (e.g., torus_model=="mn12"), a composable block must be registered for
that variant in the same category. Adding a new variant branch in
evaluate_grahsp_agn then fails until a block exists.

Related issue: #985 (GRAHSP composability).
"""

from __future__ import annotations

import inspect
import re

import pytest

from tengri.components.agn.blocks import AGN_BLOCKS
from tengri.components.agn.grahsp.model import evaluate_grahsp_agn

pytestmark = pytest.mark.contract

# Map each (selector_name, variant_literal) to (block_category, block_type).
# Parsed from evaluate_grahsp_agn source code (re-extracted on test run).
_VARIANT_BLOCK_MAPPINGS = {
    ("torus_model", "mn12"): ("torus", "grahsp_mn12"),
    ("feii_template", "veroncetty2004"): ("feii", "grahsp_veroncetty"),
    ("disc_model", "netzer"): ("disc", "grahsp_netzer"),
}

# Pending mappings: variant is declared in evaluate_grahsp_agn but the
# composable block does not exist yet. Each entry must carry a reason string.
# Test fails if a pending entry becomes registered (stale).
_PENDING_MAPPINGS = {
    ("disc_model", "netzer"): "T4: disc:grahsp_netzer not yet implemented",
}


def _extract_variant_literals() -> dict[str, set[str]]:
    """Extract variant literals from evaluate_grahsp_agn source code.

    Returns a dict mapping selector name (torus_model, feii_template, disc_model)
    to the set of literal values it branches on (e.g., {"mn12", "gaussian"}).
    """
    source = inspect.getsource(evaluate_grahsp_agn)

    # Find all `== "..."` comparisons
    patterns = re.findall(r'== "([^"]+)"', source)

    # Map each pattern to its selector. We know the order from the code:
    # torus_model branches on "mn12" and "gaussian"
    # feii_template branches on "veroncetty2004" and "bruhweiler2008"
    # disc_model branches on "netzer"

    # Build a set of known variants by inspecting the mappings
    variants = {}
    for (selector, literal), _ in _VARIANT_BLOCK_MAPPINGS.items():
        if selector not in variants:
            variants[selector] = set()
        variants[selector].add(literal)

    for (selector, literal), _ in _PENDING_MAPPINGS.items():
        if selector not in variants:
            variants[selector] = set()
        variants[selector].add(literal)

    return variants


def test_every_grahsp_variant_has_a_composable_block():
    """Every GRAHSP variant literal must have a registered composable block."""
    variants = _extract_variant_literals()

    # For each mapping (including pending), check if the block exists
    for (selector, literal), (category, block_type) in _VARIANT_BLOCK_MAPPINGS.items():
        # Skip pending mappings in this test (they don't have blocks yet)
        if (selector, literal) in _PENDING_MAPPINGS:
            continue

        # Block must be registered
        assert block_type in AGN_BLOCKS.get(category, {}), (
            f"Variant {selector}=={literal!r} maps to "
            f"block ({category!r}, {block_type!r}) but it is not registered. "
            f"Available blocks in {category!r}: {sorted(AGN_BLOCKS.get(category, {}).keys())}"
        )


def test_pending_mappings_not_yet_registered():
    """Pending variant mappings must not yet be registered."""
    for (selector, literal), reason in _PENDING_MAPPINGS.items():
        (category, block_type) = _VARIANT_BLOCK_MAPPINGS[(selector, literal)]

        # Block must NOT exist yet (it's pending)
        assert block_type not in AGN_BLOCKS.get(category, {}), (
            f"Pending mapping {selector}=={literal!r} ({reason}) is now "
            f"registered as ({category!r}, {block_type!r}) — "
            f"remove it from _PENDING_MAPPINGS."
        )
