# SPDX-License-Identifier: BSD-3-Clause
"""Contract: CalibrationELineMarginalizedLikelihood states the exact joint marginal.

The calibration polynomial and the emission-line amplitudes are marginalized in one
closed-form joint. The docstring must say so, and must not describe a sequential
point-estimate composition.

Ref: #2354
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract


def _docstring() -> str:
    from tengri.inference.likelihoods import CalibrationELineMarginalizedLikelihood

    doc = inspect.getdoc(CalibrationELineMarginalizedLikelihood)
    assert doc is not None
    return doc.lower()


class TestCalibrationELineMarginalizedLikelihoodJoint:
    """Contract test on the docstring of CalibrationELineMarginalizedLikelihood."""

    def test_docstring_states_the_exact_joint_marginal(self):
        doc = _docstring()
        assert "exact" in doc
        assert "joint" in doc

    @pytest.mark.parametrize("phrase", ["plug-in", "plug in", "point estimate"])
    def test_docstring_does_not_describe_a_plugin(self, phrase):
        assert phrase not in _docstring()
