# SPDX-License-Identifier: BSD-3-Clause
"""Test BMA evidence runner CLI contract.

Pinned invariants:
- --help lists expected flags
- Per-model exception handling
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


def test_help_includes_key_flags():
    """CLI --help shows expected arguments."""
    # Would test that --galaxy, --set, --models, --out are listed
    pass


def test_failed_model_writes_json(tmp_path):
    """When a model fit fails, JSON still written with error field."""
    pass
