# SPDX-License-Identifier: BSD-3-Clause
"""Engagement evidence (#3505 X1): every gate records its outcome, with a reason."""

from __future__ import annotations

import pytest

from tengri.forward.precompute.band_table import Decline


@pytest.mark.parametrize("bad", [None, "", "   ", 3])
def test_decline_requires_a_reason(bad):
    with pytest.raises((TypeError, ValueError)):
        Decline(bad)


def test_decline_keeps_its_reason():
    assert Decline("why").reason == "why"
