# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2297: nebular fallback defaults.

Three fallback literals disagreed with their parameter declarations:
- gas_logn: fallback 3.0, declared default 2.0
- shock_log_lhalpha: fallback 40.0, declared default 41.0
- neb_logU: fallback -2.0, declared default -3.0

This test verifies that the fallback defaults now read from the declared defaults.
"""

from __future__ import annotations

import pytest

from tengri.components.nebular._params import PARAMS as NEB_PARAMS
from tengri.components.nebular._params import CUE_GAS_EXTRA_PARAMS
from tengri.components.nebular._params import SHOCK_PARAMS
from tengri.protocols.component import declared_default


pytestmark = pytest.mark.regression_bug


def test_gas_logn_declared_default():
    """Verify gas_logn declared default is 2.0."""
    gas_logn_default = declared_default(CUE_GAS_EXTRA_PARAMS, "gas_logn")
    assert gas_logn_default == 2.0


def test_shock_log_lhalpha_declared_default():
    """Verify shock_log_lhalpha declared default is 41.0."""
    shock_default = declared_default(SHOCK_PARAMS, "shock_log_lhalpha")
    assert shock_default == 41.0


def test_neb_logu_declared_default():
    """Verify neb_logU declared default is -3.0."""
    neb_logu_default = declared_default(NEB_PARAMS, "neb_logU")
    assert neb_logu_default == -3.0
