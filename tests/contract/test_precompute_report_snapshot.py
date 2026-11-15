# SPDX-License-Identifier: BSD-3-Clause
"""The precompute engagement summary is pinned for the six band-table parity cases.

``PrecomputeEngagementReport.summary_text()`` is user-facing: it names which mechanisms engaged
and why the others declined. The band-table refactor moves each builder's cache into a
per-component memo and changes what the report reads, so the text must not move. Each golden
summary was captured on the tree before that refactor (``tests/fixtures/band_table_parity``).
"""

from __future__ import annotations

import pytest

from tengri.forward.precompute_report import precompute_engagement_report
from tests.contract import _band_table_cases as cases
from tests.contract._band_table_capture import load_golden

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("name", list(cases.CASES))
def test_summary_text_matches_golden(name):
    """``summary_text()`` of each parity case is byte-identical to its golden."""
    record, _ = load_golden(name)
    model = cases.CASES[name]()
    # The report reads the caches the template build fills, so the golden is taken after it.
    model._template_data_for_jit()
    assert precompute_engagement_report(model).summary_text() == record["summary"]
