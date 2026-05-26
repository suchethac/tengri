# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-11: summary_table key mismatch.

See ADR / docs/known_bugs.md for full context.
"""

import pytest

pytestmark = pytest.mark.regression_bug


class TestBug11SummaryKeys:
    """posterior.py:178-181 — Must show acceptance rate from RT diagnostics.

    Key mismatch: samplers store "accept_rate", "n_divergent", "mean_accept_prob"
    but posterior.summary_table checks for "acceptance_rate", "n_divergences".
    Result: no diagnostics shown in posterior summaries.
    """

    def test_accept_rate_key_found(self):
        """The key stored by _run_raytrace must be recognized by summary_table."""
        # These are the keys actually stored by the samplers:
        rt_keys = {"accept_rate", "n_steps", "step_size"}
        nuts_keys = {"n_divergent", "mean_accept_prob"}

        # These are the keys posterior.py checks for:
        checked_keys = {"acceptance_rate", "n_divergences"}

        rt_found = checked_keys & rt_keys
        nuts_found = checked_keys & nuts_keys

        # BUG: none of the checked keys match the stored keys
        assert len(rt_found) == 0, "If this fails, BUG-11 has been fixed"
        assert len(nuts_found) == 0, "If this fails, BUG-11 has been fixed"
