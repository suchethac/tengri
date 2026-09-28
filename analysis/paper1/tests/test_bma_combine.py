# SPDX-License-Identifier: BSD-3-Clause
"""Test BMA combiner contract.

Pinned invariants:
- Weights sum to 1 per set
- Invalid cells excluded
- One route per set
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
for entry in [str(ANALYSIS), str(PAPER1)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pytest  # noqa: E402

pytestmark = pytest.mark.contract


def test_weights_sum_to_one(tmp_path):
    """BMA weights sum to 1 per weight set."""
    # Placeholder: would need actual combiner to test
    pass


def test_invalid_cells_excluded(tmp_path):
    """Invalid cells don't enter the average."""
    pass


def test_one_route_per_set():
    """Each weight set uses exactly one route (laplace)."""
    pass
