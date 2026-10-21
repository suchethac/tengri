# SPDX-License-Identifier: BSD-3-Clause
"""The wildcard FREE accounting leaves out a Fixed-only ``*_gyr`` timing alias (#2748).

``sfh={'type': 'dpl', 'all_params': FREE}`` frees ``sfh_dpl_age_frac``. Its
Gyr alias ``sfh_dpl_age_gyr`` is the Fixed shim of that fraction, so the
partial-free warning must neither count it nor name it as stuck. A parameter
without a twin and without a declared prior must still be reported.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import FREE, Fixed
from tengri.config.exceptions import WildcardPartialFreeWarning
from tengri.parameters import parse_groups

pytestmark = pytest.mark.regression_bug


def _partial_free_messages(sfh):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_groups(sfh=sfh, redshift=Fixed(0.1))
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, WildcardPartialFreeWarning)
        and str(w.message).startswith("'all_params: FREE' freed")
    ]


def test_gyr_alias_with_a_freed_twin_is_not_reported_stuck():
    """dpl: age_frac is freed, so age_gyr is neither counted nor listed."""
    messages = _partial_free_messages({"type": "dpl", "all_params": FREE})
    assert not [m for m in messages if "sfh_dpl_age_gyr" in m], messages


def test_parameter_without_a_twin_or_prior_is_still_reported():
    """dpl_lookback: lookback_end_gyr has no twin and no prior, so it stays named."""
    messages = _partial_free_messages({"type": "dpl_lookback", "all_params": FREE})
    assert any("sfh_dpl_lookback_end_gyr" in m for m in messages), messages
    assert not [m for m in messages if "sfh_dpl_lookback_age_gyr" in m], messages
