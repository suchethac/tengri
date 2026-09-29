# SPDX-License-Identifier: BSD-3-Clause
"""Contract: FeaturePrecomp docstring states the line-flux accuracy it was measured to.

The line LUT reproduces measured line fluxes to 4.8e-05 – 1.0e-03 relative
(five thetas, wNE grid, #2376). This accuracy is not obvious from the
implementation, so it must be stated in the class docstring where a user
chooses the approximation.
"""

import inspect
import re

import pytest

from tengri.forward.sed_model import FeaturePrecomp

pytestmark = pytest.mark.contract


def test_feature_precomp_docstring_contains_stated_line_accuracy() -> None:
    """FeaturePrecomp docstring mentions line-flux accuracy with order of magnitude."""
    doc = inspect.getdoc(FeaturePrecomp)
    assert doc is not None

    # Search for a statement about line-flux accuracy with an order of magnitude.
    # Valid patterns: 1e-4, 1e-3, 10^-4, 10^-3, etc.
    pattern = (
        r"(line\s+fluxes?.*?(1e-[0-4]|1e-[0-3]|10\^-[0-4]|10\^-[0-3])|"
        r"1e-[0-4]|1e-[0-3]|10\^-[0-4]|10\^-[0-3].*?line\s+fluxes?)"
    )
    match = re.search(pattern, doc, re.IGNORECASE | re.DOTALL)

    assert match is not None, (
        "FeaturePrecomp docstring must state the line-flux accuracy "
        "with an order of magnitude (e.g., 1e-4, 1e-3). "
        f"Current docstring:\n{doc}"
    )

    # Verify the docstring also cites the source (issue number or bench/ path).
    source_pattern = r"#2376|bench/"
    source_match = re.search(source_pattern, doc)
    assert source_match is not None, (
        "FeaturePrecomp docstring must cite the measurement source (#2376 or bench/)"
    )
