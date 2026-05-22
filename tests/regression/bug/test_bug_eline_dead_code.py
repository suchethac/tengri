# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for eline_priors.py dead code removed bug.

Bug: eline_priors.py:251 — orphaned design_matrix.shape[1] expression (no assignment).
"""

import pytest

pytestmark = pytest.mark.regression_bug


class TestElinePriorNoDead:
    """Bug: eline_priors.py:251 — orphaned design_matrix.shape[1] expression."""

    def test_no_dead_shape_expression(self):
        """The file should not contain the bare dead-code expression."""
        import inspect

        from tengri.observation import eline_priors

        src = inspect.getsource(eline_priors)
        # The dead-code line was 'design_matrix.shape[1]' with no assignment
        assert "design_matrix.shape[1]\n" not in src, (
            "Dead code 'design_matrix.shape[1]' (no assignment) still present"
        )
