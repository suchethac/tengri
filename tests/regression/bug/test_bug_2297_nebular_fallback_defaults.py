# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2297: nebular fallback defaults.

Verifies that fallback defaults read from parameter declarations:
- AGN NLR gas_logn → declared_default(AGN_PARAMS, "agn_nlr_logn") returns 3.0
- AGN NLR neb_logU → declared_default(AGN_PARAMS, "agn_nlr_logU") returns -2.0
"""

from __future__ import annotations

import pytest

from tengri.components.agn._params import PARAMS as AGN_PARAMS
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.regression_bug


def test_agn_nlr_logn_declared_default():
    """Verify agn_nlr_logn declared default is 3.0."""
    agn_nlr_logn_default = declared_default(AGN_PARAMS, "agn_nlr_logn")
    assert agn_nlr_logn_default == 3.0


def test_agn_nlr_logu_declared_default():
    """Verify agn_nlr_logU declared default is -2.0."""
    agn_nlr_logu_default = declared_default(AGN_PARAMS, "agn_nlr_logU")
    assert agn_nlr_logu_default == -2.0
