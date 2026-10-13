# SPDX-License-Identifier: BSD-3-Clause
"""The PSB two-step metallicity fallback uses the declared burst age.

Bug: ``components/stellar/component.py`` read the burst onset with
``params.get("sfh_psb_burstage_gyr", 1.0)``. The SFH declaration in
``components/stellar/sfh/registry.py`` is ``Uniform(0.01, 5.0, default=0.5)``,
so a caller that leaves the key out got a burst age twice the declared default
and a silently different metallicity step, with no error.
"""

from __future__ import annotations

import pytest

from tengri.components.stellar.component import _PSB_BURSTAGE_DEFAULT_GYR
from tengri.components.stellar.sfh.registry import SFH_REGISTRY

pytestmark = pytest.mark.regression_bug


def test_psb_burstage_fallback_is_the_declared_default():
    """The fallback equals the ``sfh_psb_burstage_gyr`` default (0.5 Gyr), not 1.0."""
    declared = SFH_REGISTRY["psb"].params["sfh_psb_burstage_gyr"].default.default
    assert declared == pytest.approx(0.5)
    assert pytest.approx(declared) == _PSB_BURSTAGE_DEFAULT_GYR
