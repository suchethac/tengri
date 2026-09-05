# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for eline_priors.py dead code removed bug.

Bug: eline_priors.py:251 — orphaned design_matrix.shape[1] expression (no assignment).
"""

import pytest

pytestmark = pytest.mark.regression_bug


class TestElinePriorNoDead:
    """Bug: eline_priors.py:251 — orphaned design_matrix.shape[1] expression."""

    def test_no_dead_shape_expression(self):
        """The orphaned expression 'design_matrix.shape[1]' (no assignment) is gone.

        **Source-spelling assertion: no behavioral test possible.**

        Regression: eline_priors.py:251 contained the line:
            design_matrix.shape[1]

        This is dead code — the expression computes a value but assigns it nowhere,
        so it has no observable effect on program behavior. Removing dead code that
        computes and discards a value cannot break any software unless it had a
        side effect (e.g., array allocation under memory pressure, or a __getitem__
        hook that mutates state). Shape indexing has neither.

        Therefore, no behavioral test exists. This assertion guards against
        accidental re-introduction of the dead line via copy-paste. It is not
        a stand-in for behavioral testing and makes no claim about the behavior
        of eline_priors; it checks only that the specific orphaned code is absent.
        """
        import inspect

        from tengri.observation import eline_priors

        src = inspect.getsource(eline_priors)
        # The dead-code line was 'design_matrix.shape[1]' with no assignment
        assert "design_matrix.shape[1]\n" not in src, (
            "Dead code 'design_matrix.shape[1]' (no assignment) still present"
        )
