# SPDX-License-Identifier: BSD-3-Clause
r"""
Regression test for issue #2158: agn_grahsp_a_bc parameter description.

The parameter declaration states "relative to the powerlaw at 3000 nm", but
the implementation (balmer.py lines 155-156) and upstream GRAHSP both normalize
at 5100 Å (510 nm). The Balmer continuum is identically zero at 3000 nm, so
the stated reference wavelength is impossible.

This test verifies that the description correctly names the 5100 Å anchor
and cross-checks it against the implementation constant LAMBDA_5100_NM.

See https://github.com/Sucheth-Cooray/tengri/issues/2158.
"""

import re

import pytest

from tengri.components.agn.grahsp.bbb import LAMBDA_5100_NM
from tengri.parameters.registry import registry


@pytest.mark.regression_bug
def test_agn_grahsp_a_bc_description_names_correct_anchor():
    """The description must name 5100 Å and not 3000 nm."""
    record = registry().get("agn_grahsp_a_bc")
    assert record is not None, "agn_grahsp_a_bc not found in registry"

    desc = record.description
    assert "3000 nm" not in desc, f"Description incorrectly names 3000 nm. Got: {desc!r}"
    assert "3000 Å" not in desc, f"Description incorrectly names 3000 Å. Got: {desc!r}"

    # Check that the correct anchor is mentioned
    assert "5100" in desc or "510" in desc, (
        f"Description does not name the 5100 Å (510 nm) anchor. Got: {desc!r}"
    )


@pytest.mark.regression_bug
def test_agn_grahsp_a_bc_description_matches_implementation_constant():
    """The wavelength in the description must match LAMBDA_5100_NM constant."""
    record = registry().get("agn_grahsp_a_bc")
    assert record is not None, "agn_grahsp_a_bc not found in registry"

    desc = record.description

    # Extract the wavelength from the description
    # Look for patterns like "5100" (Å) or "510" (nm)
    angstrom_match = re.search(r"(\d+)\s*(?:Å|Angstrom)", desc)
    nm_match = re.search(r"(\d+)\s*nm", desc)

    anchor_nm = None
    if angstrom_match:
        angstrom_value = float(angstrom_match.group(1))
        anchor_nm = angstrom_value / 10.0
    elif nm_match:
        anchor_nm = float(nm_match.group(1))

    assert anchor_nm is not None, f"Could not extract wavelength from description. Got: {desc!r}"

    assert abs(anchor_nm - LAMBDA_5100_NM) < 0.1, (
        f"Description names {anchor_nm} nm, but LAMBDA_5100_NM = {LAMBDA_5100_NM} nm. "
        f"Description: {desc!r}"
    )
