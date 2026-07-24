# SPDX-License-Identifier: BSD-3-Clause
"""Tests for issue #1307: eight component-declared params missing from registry.

Regression test for parameters declared by components but not registered in
describe_parameter, causing KeyError on lookup.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_bug

MISSING = [
    "agn_xray_gamma",
    "agn_xray_delta_alpha_ox",
    "agn_xray_e_cut",
    "agn_T_max",
    "xray_det_hmxb",
    "xray_det_lmxb",
    "agn_polar_temperature",
    "agn_delta",
]


@pytest.mark.parametrize("name", MISSING)
def test_describe_parameter_knows_component_params(name):
    """Verify describe_parameter returns registered entries for all 8 params."""
    import tengri

    info = tengri.describe_parameter(name)  # must not raise (was KeyError)
    assert info  # non-empty description
    assert info.units  # must have units field (non-empty per registry convention)
