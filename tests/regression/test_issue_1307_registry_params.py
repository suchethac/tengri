# SPDX-License-Identifier: BSD-3-Clause
"""Tests for issue #1307: seven component-declared params missing from registry.

Regression test for parameters declared by components but not registered in
describe_parameter, causing KeyError on lookup.

``agn_polar_temperature`` removed (Task 16, item 10): it was retired (no
consumer remained in src/, grep-verified -- Task 13/14 already renamed
every real consumer to the canonical ``agn_polar_T``) and is now a soft
alias in ``parameters/_aliases.py``, not a directly registered parameter --
``describe_parameter`` does not consult that alias table (it is consulted
only at ``parse_groups`` build time), so asserting it here would now
legitimately raise ``KeyError``, a different failure mode than #1307's
"declared but not registered". #1307's OWN regression -- that a genuinely
component-declared parameter is describable -- is still guarded by the
other seven entries.
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
    "agn_delta",
]


@pytest.mark.parametrize("name", MISSING)
def test_describe_parameter_knows_component_params(name):
    """Verify describe_parameter returns registered entries for all 7 params."""
    import tengri

    info = tengri.describe_parameter(name)  # must not raise (was KeyError)
    assert info  # non-empty description
    assert info.units  # must have units field (non-empty per registry convention)
